import argparse
import io
import json
import math
import os
import platform
import re
import struct
import sys
import time
from pathlib import Path


CHECKPOINT_ID = "facebook/map-anything-apache"
MAPANYTHING_COMMIT = "c845b8f4f6cde0c20aecd87573656c3f69f5b2b0"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_BLUR_THRESHOLD = 1.5
DEFAULT_DUPLICATE_HAMMING_THRESHOLD = 4
DEFAULT_CONFIDENCE_PERCENTILE = 10.0
DEFAULT_MAX_OUTPUT_VIEWS = 10


def pixel_center_uv(indices, height, width) -> "numpy.ndarray":
    import numpy as np

    for name, value in (("height", height), ("width", width)):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer")

    indices = np.asarray(indices)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("indices must be a non-empty one-dimensional array")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("indices must contain integers")
    if indices.size > 1 and np.any(indices[1:] <= indices[:-1]):
        raise ValueError("indices must be strictly increasing")
    if np.any(indices < 0) or np.any(indices >= height * width):
        raise ValueError("indices are outside the image bounds")

    rows, columns = np.divmod(indices, width)
    uv = np.column_stack(
        ((columns + 0.5) / width, 1.0 - (rows + 0.5) / height)
    )
    if not np.isfinite(uv).all() or not ((uv > 0) & (uv < 1)).all():
        raise ValueError("UV coordinates must be finite and inside (0, 1)")
    return uv


def build_textured_scene(world_points, images, masks) -> tuple["trimesh.Scene", dict]:
    import numpy as np
    import trimesh
    from mapanything.utils.hf_utils.viz import image_mesh
    from PIL import Image

    world_points = np.asarray(world_points)
    images = np.asarray(images)
    masks = np.asarray(masks)
    if world_points.ndim != 4 or world_points.shape[-1] != 3:
        raise ValueError("world_points must have shape (views, height, width, 3)")

    view_count, height, width, _ = world_points.shape
    if view_count == 0:
        raise ValueError("at least one view is required")
    if masks.shape != (view_count, height, width):
        raise ValueError("masks must match the world-point grids")
    if images.shape == (view_count, height, width, 3):
        frame_images = images
    elif images.shape == (view_count, 3, height, width):
        frame_images = np.transpose(images, (0, 2, 3, 1))
    else:
        raise ValueError("images must be NHWC or NCHW and match the world-point grids")

    scene = trimesh.Scene()
    coordinate_flip = np.array([1, -1, 1], dtype=np.float32)
    for frame_index, (frame_points, frame_image, frame_mask) in enumerate(
        zip(world_points, frame_images, masks)
    ):
        faces, vertices, _colors, indices = image_mesh(
            frame_points * coordinate_flip,
            frame_image,
            mask=frame_mask,
            tri=True,
            return_indices=True,
        )
        if len(faces) == 0:
            raise ValueError(f"view {frame_index} has no valid faces")

        uv = pixel_center_uv(indices, height, width)
        if len(vertices) != len(indices) or len(vertices) != len(uv):
            raise ValueError(f"view {frame_index} vertex, index, and UV counts differ")

        texture = Image.fromarray(
            np.clip(frame_image * 255, 0, 255).astype(np.uint8)
        )
        texture.format = "JPEG"
        material = trimesh.visual.material.PBRMaterial(
            name=f"view_{frame_index:04d}_material",
            baseColorTexture=texture,
            metallicFactor=0.0,
            roughnessFactor=1.0,
            doubleSided=True,
        )
        visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
        name = f"view_{frame_index:04d}"
        scene.add_geometry(
            trimesh.Trimesh(
                vertices=vertices * coordinate_flip,
                faces=faces,
                visual=visual,
                process=False,
            ),
            geom_name=name,
            node_name=name,
        )

    rotation_matrix_x = trimesh.transformations.rotation_matrix(
        np.pi, [1, 0, 0]
    )
    scene.apply_transform(rotation_matrix_x)
    return scene, {
        "texture_mode": "embedded_jpeg_per_view",
        "mesh_count": view_count,
        "material_count": view_count,
        "texture_count": view_count,
        "image_count": view_count,
        "texture_width": width,
        "texture_height": height,
        "glb_bytes": None,
    }


def inspect_textured_glb(path, expected_views) -> dict:
    from PIL import Image

    if isinstance(expected_views, bool) or not isinstance(expected_views, int):
        raise ValueError("expected_views must be a positive integer")
    if expected_views <= 0:
        raise ValueError("expected_views must be a positive integer")

    raw = Path(path).read_bytes()
    if len(raw) < 12:
        raise ValueError("GLB header is truncated")
    magic, version, declared_length = struct.unpack_from("<4sII", raw)
    if (
        magic != b"glTF"
        or version != 2
        or declared_length != len(raw)
        or declared_length % 4 != 0
    ):
        raise ValueError("invalid GLB magic, version, file length, or alignment")

    chunks = []
    offset = 12
    while offset < len(raw):
        if offset % 4 != 0 or offset + 8 > len(raw):
            raise ValueError("GLB chunk header is truncated")
        chunk_length, chunk_type = struct.unpack_from("<II", raw, offset)
        chunk_start = offset + 8
        chunk_end = chunk_start + chunk_length
        if chunk_length % 4 != 0 or chunk_start % 4 != 0:
            raise ValueError("GLB chunk length or alignment is invalid")
        if chunk_end > len(raw):
            raise ValueError("GLB chunk exceeds the file length")
        chunks.append((chunk_type, raw[chunk_start:chunk_end]))
        offset = chunk_end
    if len(chunks) != 2 or [chunk[0] for chunk in chunks] != [
        0x4E4F534A,
        0x004E4942,
    ]:
        raise ValueError("GLB must contain one JSON chunk followed by one BIN chunk")

    document = json.loads(chunks[0][1].decode("utf-8"))
    binary = chunks[1][1]
    collections = {}
    for key in ("meshes", "materials", "textures", "images"):
        values = document.get(key)
        if not isinstance(values, list) or len(values) != expected_views:
            raise ValueError(f"GLB must contain {expected_views} {key}")
        collections[key] = values

    material_references = []
    for mesh in collections["meshes"]:
        primitives = mesh.get("primitives")
        if not isinstance(primitives, list) or len(primitives) != 1:
            raise ValueError("each mesh must contain exactly one primitive")
        primitive = primitives[0]
        attributes = primitive.get("attributes", {})
        if not isinstance(attributes, dict) or set(attributes) != {
            "POSITION",
            "TEXCOORD_0",
        }:
            raise ValueError(
                "each primitive must contain exactly POSITION and TEXCOORD_0"
            )
        material = primitive.get("material")
        if not isinstance(material, int) or not 0 <= material < expected_views:
            raise ValueError("each primitive must reference a valid material")
        material_references.append(material)
    if sorted(material_references) != list(range(expected_views)):
        raise ValueError("each material must be referenced exactly once")

    texture_references = []
    for material in collections["materials"]:
        pbr = material.get("pbrMetallicRoughness", {})
        base_color = pbr.get("baseColorTexture", {})
        texture = base_color.get("index")
        if not isinstance(texture, int) or not 0 <= texture < expected_views:
            raise ValueError("each material must reference a valid base-color texture")
        texture_references.append(texture)
    if sorted(texture_references) != list(range(expected_views)):
        raise ValueError("each texture must be referenced exactly once")

    image_references = []
    for texture in collections["textures"]:
        image = texture.get("source")
        if not isinstance(image, int) or not 0 <= image < expected_views:
            raise ValueError("each texture must reference a valid image")
        image_references.append(image)
    if sorted(image_references) != list(range(expected_views)):
        raise ValueError("each image must be referenced exactly once")

    buffers = document.get("buffers")
    buffer_views = document.get("bufferViews")
    if (
        not isinstance(buffers, list)
        or len(buffers) != 1
        or not isinstance(buffers[0], dict)
        or "uri" in buffers[0]
    ):
        raise ValueError("GLB binary data must be embedded")
    buffer_length = buffers[0].get("byteLength")
    if type(buffer_length) is not int or buffer_length < 0:
        raise ValueError("GLB buffer byteLength must be a non-negative integer")
    padding_length = len(binary) - buffer_length
    binary_padding = binary[buffer_length:]
    if not 0 <= padding_length <= 3 or any(binary_padding):
        raise ValueError("GLB buffer length permits only 0-3 zero BIN padding bytes")
    if not isinstance(buffer_views, list):
        raise ValueError("GLB bufferViews are missing")
    for view in buffer_views:
        if not isinstance(view, dict) or type(view.get("buffer")) is not int:
            raise ValueError("each bufferView must explicitly reference buffer 0")
        if view["buffer"] != 0:
            raise ValueError("each bufferView must explicitly reference buffer 0")
        start = view.get("byteOffset", 0)
        length = view.get("byteLength")
        if (
            type(start) is not int
            or type(length) is not int
            or start < 0
            or start % 4 != 0
            or length <= 0
            or start + length > buffer_length
        ):
            raise ValueError(
                "each bufferView must be aligned and inside the declared buffer"
            )

    dimensions = set()
    for image in collections["images"]:
        view_index = image.get("bufferView")
        if image.get("mimeType") != "image/jpeg" or "uri" in image:
            raise ValueError("each image must be an embedded JPEG")
        if not isinstance(view_index, int) or not 0 <= view_index < len(buffer_views):
            raise ValueError("each image must reference a valid bufferView")
        view = buffer_views[view_index]
        start = view.get("byteOffset", 0)
        length = view["byteLength"]
        image_bytes = binary[start : start + length]
        if image_bytes[:3] != b"\xff\xd8\xff":
            raise ValueError("embedded image does not have JPEG magic")
        with Image.open(io.BytesIO(image_bytes)) as decoded:
            decoded.load()
            if decoded.format != "JPEG" or decoded.width <= 0 or decoded.height <= 0:
                raise ValueError("embedded JPEG dimensions are invalid")
            dimensions.add(decoded.size)
    if len(dimensions) != 1:
        raise ValueError("embedded JPEG dimensions must be consistent")

    texture_width, texture_height = dimensions.pop()
    return {
        "texture_mode": "embedded_jpeg_per_view",
        "mesh_count": expected_views,
        "material_count": expected_views,
        "texture_count": expected_views,
        "image_count": expected_views,
        "texture_width": texture_width,
        "texture_height": texture_height,
        "glb_bytes": len(raw),
    }


def export_scene_atomic(
    scene,
    pending_asset: dict,
    output_dir: Path,
    expected_views: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = output_dir / "scene.glb.tmp"
    final_path = output_dir / "scene.glb"
    try:
        scene.export(temporary_path, file_type="glb")
        asset = inspect_textured_glb(temporary_path, expected_views=expected_views)
        if (asset["texture_width"], asset["texture_height"]) != (
            pending_asset["texture_width"],
            pending_asset["texture_height"],
        ):
            raise ValueError("exported JPEG dimensions differ from the model images")
        temporary_path.replace(final_path)
        return asset
    finally:
        temporary_path.unlink(missing_ok=True)


def sharpness_score(gray) -> float:
    import numpy as np

    gray = np.asarray(gray, dtype=np.float32)
    if gray.ndim != 2 or min(gray.shape) < 3:
        raise ValueError("gray image must be a two-dimensional array of at least 3x3")
    center = gray[1:-1, 1:-1]
    up = gray[:-2, 1:-1]
    down = gray[2:, 1:-1]
    left = gray[1:-1, :-2]
    right = gray[1:-1, 2:]
    laplacian = 4 * center - up - down - left - right
    gx = 0.5 * (right - left)
    gy = 0.5 * (down - up)
    gradient_energy = 0.5 * (np.mean(gx * gx) + np.mean(gy * gy))
    score = float(np.var(laplacian) / (gradient_energy + 1e-12))
    if not math.isfinite(score):
        raise ValueError("invalid_sharpness")
    return score


def dhash64(gray) -> int:
    import numpy as np
    from PIL import Image

    resized = gray.resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(resized, dtype=np.uint8)
    value = 0
    for bit in (pixels[:, 1:] > pixels[:, :-1]).reshape(-1):
        value = (value << 1) | int(bit)
    return value


def analyze_image_quality(
    path: Path, *, enabled: bool
) -> tuple[float | None, int | None]:
    import numpy as np
    from PIL import Image, ImageOps

    try:
        with Image.open(path) as source:
            source.load()
            oriented = ImageOps.exif_transpose(source)
    except OSError as error:
        raise ValueError(f"{path.name}: cannot decode image: {error}") from error
    if min(oriented.size) < 3:
        raise ValueError(f"{path.name}: image_too_small")
    if not enabled:
        return None, None
    gray = oriented.convert("L")
    gray.thumbnail((256, 256), Image.Resampling.LANCZOS)
    score = sharpness_score(np.asarray(gray, dtype=np.float32) / 255.0)
    return score, dhash64(gray)


def filter_input_images(
    images: list[Path],
    *,
    mode: str,
    blur_threshold: float,
    duplicate_hamming_threshold: int,
) -> tuple[list[Path], list[dict]]:
    if mode not in {"balanced", "off"}:
        raise ValueError("quality filter mode must be balanced or off")

    records = []
    for index, path in enumerate(images):
        score, image_hash = analyze_image_quality(path, enabled=mode == "balanced")
        records.append(
            {
                "index": index,
                "path": path,
                "sharpness": score,
                "hash": image_hash,
                "decision": {
                    "filename": path.name,
                    "sharpness": score,
                    "accepted_for_inference": mode == "off",
                    "rejection_reason": None,
                    "near_duplicate_of": None,
                },
            }
        )
    if mode == "off":
        return list(images), [record["decision"] for record in records]

    groups = []
    for record in records:
        if record["sharpness"] < blur_threshold:
            record["decision"]["rejection_reason"] = "blurry"
            continue
        matches = []
        for group in groups:
            distance = (record["hash"] ^ group["anchor_hash"]).bit_count()
            if distance <= duplicate_hamming_threshold:
                matches.append((distance, group["anchor_index"], group))
        if matches:
            min(matches, key=lambda item: (item[0], item[1]))[2]["members"].append(
                record
            )
        else:
            groups.append(
                {
                    "anchor_hash": record["hash"],
                    "anchor_index": record["index"],
                    "members": [record],
                }
            )

    for group in groups:
        winner = max(
            group["members"],
            key=lambda record: (record["sharpness"], -record["index"]),
        )
        winner["decision"]["accepted_for_inference"] = True
        for record in group["members"]:
            if record is winner:
                continue
            record["decision"]["rejection_reason"] = "near_duplicate"
            record["decision"]["near_duplicate_of"] = winner["path"].name

    accepted = [
        record["path"]
        for record in records
        if record["decision"]["accepted_for_inference"]
    ]
    decisions = [record["decision"] for record in records]
    return accepted, decisions


def discover_images(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise ValueError(f"input directory does not exist: {input_dir}")

    def natural_key(path: Path):
        return tuple(
            int(part) if part.isdigit() else part.casefold()
            for part in re.split(r"(\d+)", path.name)
        )

    images = sorted(
        (
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=natural_key,
    )
    if len(images) < 3:
        raise ValueError("input directory must contain at least 3 JPG, JPEG, or PNG images")
    return images


def base_report(images: list[Path]) -> dict:
    return {
        "schema_version": 2,
        "status": "running",
        "mapanything_commit": MAPANYTHING_COMMIT,
        "checkpoint_id": CHECKPOINT_ID,
        "inputs": [path.name for path in images],
        "quality_filter": {},
        "environment": {},
        "timings": {},
        "views": [],
    }


def write_report(output_dir: Path, report: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = output_dir / "report.json.tmp"
    try:
        payload = json.dumps(
            report, ensure_ascii=False, indent=2, allow_nan=False
        ) + "\n"
        temporary_path.write_text(payload, encoding="utf-8")
        temporary_path.replace(output_dir / "report.json")
    finally:
        temporary_path.unlink(missing_ok=True)


def validate_options(
    *,
    mode: str,
    blur_threshold: float | str,
    duplicate_hamming_threshold: int | str,
    confidence_percentile: float | str,
    max_output_views: int | str,
) -> tuple[float, int, float, int]:
    if mode not in {"balanced", "off"}:
        raise ValueError("quality filter mode must be balanced or off")
    try:
        blur_threshold = float(blur_threshold)
        confidence_percentile = float(confidence_percentile)
    except (TypeError, ValueError) as error:
        raise ValueError("numeric quality options must contain numbers") from error

    def parse_integer(value, name: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be an integer")
        if isinstance(value, int):
            return value
        if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value):
            return int(value)
        raise ValueError(f"{name} must be an integer")

    duplicate_hamming_threshold = parse_integer(
        duplicate_hamming_threshold, "duplicate_hamming_threshold"
    )
    max_output_views = parse_integer(max_output_views, "max_output_views")
    if not math.isfinite(blur_threshold) or not 0 <= blur_threshold <= 100:
        raise ValueError("blur_threshold must be finite and between 0 and 100")
    if not 0 <= duplicate_hamming_threshold <= 16:
        raise ValueError("duplicate_hamming_threshold must be between 0 and 16")
    if not math.isfinite(confidence_percentile) or not 0 <= confidence_percentile <= 50:
        raise ValueError("confidence_percentile must be finite and between 0 and 50")
    if not 6 <= max_output_views <= 10:
        raise ValueError("max_output_views must be between 6 and 10")
    return (
        blur_threshold,
        duplicate_hamming_threshold,
        confidence_percentile,
        max_output_views,
    )


def prepare_output(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("scene.glb", "scene.glb.tmp", "report.json.tmp"):
        (output_dir / name).unlink(missing_ok=True)


def finalize_timings(timings: dict[str, float]) -> None:
    keys = (
        "quality_filter_seconds",
        "cold_load_seconds",
        "image_load_seconds",
        "inference_seconds",
        "export_seconds",
    )
    if any(
        key not in timings
        or not math.isfinite(timings[key])
        or timings[key] < 0
        for key in keys
    ):
        raise ValueError("timing values must be finite non-negative seconds")
    timings["load_seconds"] = (
        timings["cold_load_seconds"] + timings["image_load_seconds"]
    )
    timings["warm_preview_seconds"] = (
        timings["quality_filter_seconds"]
        + timings["image_load_seconds"]
        + timings["inference_seconds"]
        + timings["export_seconds"]
    )
    timings["total_seconds"] = (
        timings["cold_load_seconds"] + timings["warm_preview_seconds"]
    )


def validate_completed_report(report: dict) -> None:
    quality = report.get("quality_filter", {})
    decisions = quality.get("images", [])
    views = report.get("views", [])
    if report.get("schema_version") != 2:
        raise ValueError("completed report must use schema version 2")
    if not report.get("environment"):
        raise ValueError("completed report must contain environment data")
    mode = quality.get("mode")
    if mode not in {"balanced", "off"}:
        raise ValueError("quality filter mode must be balanced or off")
    for row in decisions:
        reason = row.get("rejection_reason")
        if reason not in {None, "blurry", "near_duplicate"}:
            raise ValueError("quality rejection reason is invalid")
        if bool(row.get("accepted_for_inference")) != (reason is None):
            raise ValueError("accepted flag and rejection reason differ")
    if quality.get("input_count") != len(report.get("inputs", [])) or len(
        decisions
    ) != quality.get("input_count"):
        raise ValueError("input counts do not match")
    if quality.get("inference_count") != sum(
        bool(row.get("accepted_for_inference")) for row in decisions
    ) or len(views) != quality.get("inference_count"):
        raise ValueError("inference counts do not match")
    usable_count = sum(bool(view.get("usable_for_export")) for view in views)
    export_count = sum(bool(view.get("exported")) for view in views)
    if quality.get("usable_view_count") != usable_count:
        raise ValueError("usable view counts do not match")
    if quality.get("export_view_count") != export_count or export_count < 3:
        raise ValueError("export view counts do not match")

    asset = report.get("asset", {})
    if any(
        asset.get(key) != export_count
        for key in ("mesh_count", "material_count", "texture_count", "image_count")
    ):
        raise ValueError("asset counts do not match exported views")
    allowed_view_reasons = {
        None,
        "no_valid_pixels_after_confidence",
        "no_valid_quad_after_confidence",
        "not_selected_as_representative",
    }
    for view in views:
        reason = view.get("exclusion_reason")
        if reason not in allowed_view_reasons:
            raise ValueError("view exclusion reason is invalid")
        if bool(view.get("exported")) != (reason is None):
            raise ValueError("exported flag and exclusion reason differ")
        if view.get("exported") and not view.get("usable_for_export"):
            raise ValueError("an unusable view cannot be exported")

    has_low_count_warning = "fewer_than_six_export_views" in quality.get(
        "warnings", []
    )
    if has_low_count_warning != (3 <= export_count < 6):
        raise ValueError("fewer-than-six warning does not match export count")
    if mode == "off" and any(
        quality.get(key) is not None
        for key in (
            "blur_threshold",
            "duplicate_hamming_threshold",
            "confidence_percentile",
            "max_output_views",
        )
    ):
        raise ValueError("off thresholds must be null")
    required_timings = (
        "quality_filter_seconds",
        "cold_load_seconds",
        "image_load_seconds",
        "inference_seconds",
        "export_seconds",
        "load_seconds",
        "warm_preview_seconds",
        "total_seconds",
    )
    if any(
        key not in report.get("timings", {})
        or not math.isfinite(report["timings"][key])
        or report["timings"][key] < 0
        for key in required_timings
    ):
        raise ValueError("completed timings must be finite non-negative seconds")
    timings = report["timings"]
    expected_load = timings["cold_load_seconds"] + timings["image_load_seconds"]
    expected_warm = (
        timings["quality_filter_seconds"]
        + timings["image_load_seconds"]
        + timings["inference_seconds"]
        + timings["export_seconds"]
    )
    if (
        timings["load_seconds"] != expected_load
        or timings["warm_preview_seconds"] != expected_warm
        or timings["total_seconds"] != timings["cold_load_seconds"] + expected_warm
    ):
        raise ValueError("completed timing formulas do not match stage timings")


def infer_views(model, views, *, mode: str, confidence_percentile: float):
    kwargs = {
        "memory_efficient_inference": True,
        "minibatch_size": 1,
        "use_amp": True,
        "amp_dtype": "bf16",
        "apply_mask": True,
        "mask_edges": True,
        "apply_confidence_mask": mode == "balanced",
        "use_multiview_confidence": False,
    }
    if mode == "balanced":
        kwargs["confidence_percentile"] = confidence_percentile
    return model.infer(views, **kwargs)


def summarize_view(mask, confidence, *, require_quad: bool = True) -> dict:
    import numpy as np

    mask = np.asarray(mask, dtype=bool)
    confidence = np.asarray(confidence, dtype=np.float32)
    if mask.ndim != 2 or confidence.shape != mask.shape:
        raise ValueError("mask and confidence must be matching two-dimensional arrays")
    valid_fraction = float(mask.mean())
    if not mask.any():
        return {
            "mean_valid_confidence": None,
            "valid_fraction": 0.0,
            "selection_score": None,
            "usable_for_export": False,
            "exported": False,
            "exclusion_reason": "no_valid_pixels_after_confidence",
        }
    valid_confidence = confidence[mask]
    if not np.isfinite(valid_confidence).all():
        raise ValueError("valid confidence values must be finite")
    mean_confidence = float(valid_confidence.mean())
    has_quad = bool(
        (
            mask[:-1, :-1]
            & mask[1:, :-1]
            & mask[:-1, 1:]
            & mask[1:, 1:]
        ).any()
    )
    if require_quad and not has_quad:
        return {
            "mean_valid_confidence": mean_confidence,
            "valid_fraction": valid_fraction,
            "selection_score": None,
            "usable_for_export": False,
            "exported": False,
            "exclusion_reason": "no_valid_quad_after_confidence",
        }
    return {
        "mean_valid_confidence": mean_confidence,
        "valid_fraction": valid_fraction,
        "selection_score": mean_confidence * valid_fraction,
        "usable_for_export": True,
        "exported": False,
        "exclusion_reason": "not_selected_as_representative",
    }


def select_representative_indices(
    scores: list[float], max_output_views: int
) -> list[int]:
    if not 6 <= max_output_views <= 10:
        raise ValueError("max_output_views must be 6 through 10")
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("selection scores must be finite")
    count = len(scores)
    if count <= max_output_views:
        return list(range(count))
    selected = []
    for bin_index in range(max_output_views):
        start = bin_index * count // max_output_views
        end = (bin_index + 1) * count // max_output_views
        selected.append(
            max(range(start, end), key=lambda index: (scores[index], -index))
        )
    return selected


def select_export_view_indices(
    view_reports: list[dict], *, mode: str, max_output_views: int
) -> list[int]:
    usable_indices = [
        index
        for index, report in enumerate(view_reports)
        if report["usable_for_export"]
    ]
    if len(usable_indices) < 3:
        raise ValueError(
            f"confidence mask left {len(usable_indices)} usable views; at least 3 are required"
        )
    relative_indices = (
        list(range(len(usable_indices)))
        if mode == "off"
        else select_representative_indices(
            [view_reports[index]["selection_score"] for index in usable_indices],
            max_output_views,
        )
    )
    selected_indices = [usable_indices[index] for index in relative_indices]
    for index in selected_indices:
        view_reports[index]["exported"] = True
        view_reports[index]["exclusion_reason"] = None
    return selected_indices


def _run_inference(
    images: list[Path],
    output_dir: Path,
    report: dict,
    *,
    mode: str,
    confidence_percentile: float,
    max_output_views: int,
) -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cold_started = time.perf_counter()

    import numpy as np
    import torch
    import trimesh  # noqa: F401
    from mapanything.models import MapAnything
    from mapanything.utils.geometry import depthmap_to_world_frame
    from mapanything.utils.hf_utils.viz import image_mesh  # noqa: F401
    from mapanything.utils.image import load_images
    from PIL import Image  # noqa: F401

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is disabled")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    model = MapAnything.from_pretrained(CHECKPOINT_ID).to(device)
    model.eval()
    torch.cuda.synchronize(device)
    cold_load_seconds = time.perf_counter() - cold_started

    image_load_started = time.perf_counter()
    views = load_images([str(path) for path in images])
    torch.cuda.synchronize(device)
    image_load_seconds = time.perf_counter() - image_load_started

    inference_started = time.perf_counter()
    outputs = infer_views(
        model,
        views,
        mode=mode,
        confidence_percentile=confidence_percentile,
    )
    torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - inference_started
    export_started = time.perf_counter()
    if len(outputs) != len(images):
        raise RuntimeError(
            f"model returned {len(outputs)} views for {len(images)} input images"
        )

    reconstructions = []
    view_reports = []
    for image_path, prediction in zip(images, outputs):
        depth = prediction["depth_z"][0].squeeze(-1)
        intrinsics = prediction["intrinsics"][0]
        camera_to_world = prediction["camera_poses"][0]
        world_points, valid_depth = depthmap_to_world_frame(
            depth, intrinsics, camera_to_world
        )
        mask = prediction["mask"][0].squeeze(-1).detach().cpu().numpy().astype(bool)
        mask &= valid_depth.detach().cpu().numpy()
        confidence = (
            prediction["conf"][0]
            .squeeze(-1)
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        summary = summarize_view(mask, confidence, require_quad=mode == "balanced")
        view_report = {
            "filename": image_path.name,
            "camera_to_world": camera_to_world.detach().cpu().tolist(),
            "intrinsics": intrinsics.detach().cpu().tolist(),
            **summary,
            "metric_scale": float(
                prediction["metric_scaling_factor"]
                .detach()
                .cpu()
                .reshape(-1)[0]
                .item()
            ),
        }
        report_index = len(view_reports)
        view_reports.append(view_report)
        if not summary["usable_for_export"]:
            if mode == "off":
                report["views"] = view_reports
                write_report(output_dir, report)
                raise ValueError(f"{image_path.name} has no valid reconstructed pixels")
            continue
        reconstructions.append(
            {
                "report_index": report_index,
                "world_points": world_points.detach().cpu().numpy(),
                "image": prediction["img_no_norm"][0].detach().cpu().numpy(),
                "mask": mask,
            }
        )

    report["views"] = view_reports
    report["quality_filter"]["usable_view_count"] = len(reconstructions)
    capability = torch.cuda.get_device_capability(device)
    report["environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_compute_capability": f"{capability[0]}.{capability[1]}",
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1048576,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1048576,
    }
    report["timings"].update(
        {
            "cold_load_seconds": cold_load_seconds,
            "image_load_seconds": image_load_seconds,
            "inference_seconds": inference_seconds,
        }
    )
    if len(reconstructions) < 3:
        report["quality_filter"]["export_view_count"] = 0
        write_report(output_dir, report)
        raise ValueError(
            f"confidence mask left {len(reconstructions)} usable views; at least 3 are required"
        )

    selected_report_indices = select_export_view_indices(
        view_reports,
        mode=mode,
        max_output_views=max_output_views,
    )
    reconstruction_by_report = {
        item["report_index"]: item for item in reconstructions
    }
    selected = [
        reconstruction_by_report[index] for index in selected_report_indices
    ]
    report["quality_filter"]["export_view_count"] = len(selected)
    if len(selected) < 6:
        report["quality_filter"]["warnings"].append(
            "fewer_than_six_export_views"
        )
    write_report(output_dir, report)

    scene, pending_asset = build_textured_scene(
        np.stack([item["world_points"] for item in selected]),
        np.stack([item["image"] for item in selected]),
        np.stack([item["mask"] for item in selected]),
    )
    asset = export_scene_atomic(
        scene,
        pending_asset,
        output_dir,
        expected_views=len(selected),
    )
    torch.cuda.synchronize(device)
    report["timings"]["export_seconds"] = time.perf_counter() - export_started
    finalize_timings(report["timings"])
    report["asset"] = asset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a MapAnything multi-image GLB reconstruction"
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--quality-filter", choices=("balanced", "off"), default="balanced")
    parser.add_argument("--blur-threshold", default=DEFAULT_BLUR_THRESHOLD)
    parser.add_argument(
        "--duplicate-hamming-threshold",
        default=DEFAULT_DUPLICATE_HAMMING_THRESHOLD,
    )
    parser.add_argument(
        "--confidence-percentile", default=DEFAULT_CONFIDENCE_PERCENTILE
    )
    parser.add_argument("--max-output-views", default=DEFAULT_MAX_OUTPUT_VIEWS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = base_report([])
    checkpoint_written = False
    try:
        prepare_output(args.output_dir)
        (
            args.blur_threshold,
            args.duplicate_hamming_threshold,
            args.confidence_percentile,
            args.max_output_views,
        ) = validate_options(
            mode=args.quality_filter,
            blur_threshold=args.blur_threshold,
            duplicate_hamming_threshold=args.duplicate_hamming_threshold,
            confidence_percentile=args.confidence_percentile,
            max_output_views=args.max_output_views,
        )
        images = discover_images(args.input_dir)
        report = base_report(images)
        quality_started = time.perf_counter()
        inference_images, decisions = filter_input_images(
            images,
            mode=args.quality_filter,
            blur_threshold=args.blur_threshold,
            duplicate_hamming_threshold=args.duplicate_hamming_threshold,
        )
        quality_seconds = time.perf_counter() - quality_started
        balanced = args.quality_filter == "balanced"
        report["quality_filter"] = {
            "mode": args.quality_filter,
            "blur_threshold": args.blur_threshold if balanced else None,
            "duplicate_hamming_threshold": (
                args.duplicate_hamming_threshold if balanced else None
            ),
            "confidence_percentile": (
                args.confidence_percentile if balanced else None
            ),
            "max_output_views": args.max_output_views if balanced else None,
            "input_count": len(images),
            "inference_count": len(inference_images),
            "usable_view_count": None,
            "export_view_count": None,
            "warnings": [],
            "images": decisions,
        }
        report["timings"]["quality_filter_seconds"] = quality_seconds
        write_report(args.output_dir, report)
        checkpoint_written = True
        if len(inference_images) < 3:
            raise ValueError(
                f"quality filter left {len(inference_images)} images; at least 3 are required"
            )

        _run_inference(
            inference_images,
            args.output_dir,
            report,
            mode=args.quality_filter,
            confidence_percentile=args.confidence_percentile,
            max_output_views=args.max_output_views,
        )
        validate_completed_report(report)
        json.dumps(report, allow_nan=False)
        report["status"] = "success"
        write_report(args.output_dir, report)
        return 0
    except Exception as error:
        try:
            prepare_output(args.output_dir)
        except Exception:
            pass
        try:
            json.dumps(report, allow_nan=False)
        except (TypeError, ValueError):
            report = None
            if checkpoint_written:
                try:
                    checkpoint = json.loads(
                        (args.output_dir / "report.json").read_text("utf-8")
                    )
                    if not isinstance(checkpoint, dict):
                        raise ValueError("report checkpoint must be an object")
                    json.dumps(checkpoint, allow_nan=False)
                    report = checkpoint
                except (OSError, TypeError, ValueError):
                    pass
            if report is None:
                report = base_report([])
        report.pop("asset", None)
        report["status"] = "failed"
        report["error"] = {
            "class": type(error).__name__,
            "message": str(error),
        }
        try:
            write_report(args.output_dir, report)
        except Exception:
            pass
        print(f"error: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
