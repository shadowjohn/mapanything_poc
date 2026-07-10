import argparse
import io
import json
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
        if "POSITION" not in attributes or "TEXCOORD_0" not in attributes:
            raise ValueError("each primitive must contain POSITION and TEXCOORD_0")
        if "COLOR_0" in attributes:
            raise ValueError("textured primitives must not contain COLOR_0")
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
        "texture_width": texture_width,
        "texture_height": texture_height,
        "glb_bytes": len(raw),
    }


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
        "schema_version": 1,
        "status": "running",
        "mapanything_commit": MAPANYTHING_COMMIT,
        "checkpoint_id": CHECKPOINT_ID,
        "inputs": [path.name for path in images],
        "environment": {},
        "timings": {},
        "views": [],
    }


def write_report(output_dir: Path, report: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = output_dir / "report.json.tmp"
    temporary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_dir / "report.json")


def _run_inference(images: list[Path], output_dir: Path) -> dict:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import numpy as np
    import torch
    from mapanything.models import MapAnything
    from mapanything.utils.geometry import depthmap_to_world_frame
    from mapanything.utils.image import load_images

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is disabled")

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    total_started = time.perf_counter()

    load_started = total_started
    # Preload one-time export dependencies during load so warm timings stay warm.
    import trimesh  # noqa: F401
    from mapanything.utils.hf_utils.viz import image_mesh  # noqa: F401
    from PIL import Image  # noqa: F401

    model = MapAnything.from_pretrained(CHECKPOINT_ID).to(device)
    model.eval()
    views = load_images([str(path) for path in images])
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - load_started

    torch.cuda.synchronize(device)
    inference_started = time.perf_counter()
    outputs = model.infer(
        views,
        memory_efficient_inference=True,
        minibatch_size=1,
        use_amp=True,
        amp_dtype="bf16",
        apply_mask=True,
        mask_edges=True,
    )
    torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - inference_started

    if len(outputs) != len(images):
        raise RuntimeError(
            f"model returned {len(outputs)} views for {len(images)} input images"
        )

    world_points_list = []
    images_list = []
    masks_list = []
    view_reports = []
    for image_path, prediction in zip(images, outputs):
        depth = prediction["depth_z"][0].squeeze(-1)
        intrinsics = prediction["intrinsics"][0]
        camera_to_world = prediction["camera_poses"][0]
        world_points, valid_depth = depthmap_to_world_frame(
            depth, intrinsics, camera_to_world
        )

        mask = prediction["mask"][0].squeeze(-1).detach().cpu().numpy().astype(bool)
        mask = mask & valid_depth.detach().cpu().numpy()
        if not mask.any():
            raise ValueError(f"{image_path.name} has no valid reconstructed pixels")

        confidence = prediction["conf"][0].detach().float().cpu().numpy()
        world_points_list.append(world_points.detach().cpu().numpy())
        images_list.append(prediction["img_no_norm"][0].detach().cpu().numpy())
        masks_list.append(mask)
        view_reports.append(
            {
                "filename": image_path.name,
                "camera_to_world": camera_to_world.detach().cpu().tolist(),
                "intrinsics": intrinsics.detach().cpu().tolist(),
                "mean_valid_confidence": float(confidence[mask].mean()),
                "valid_fraction": float(mask.mean()),
                "metric_scale": float(
                    prediction["metric_scaling_factor"]
                    .detach()
                    .cpu()
                    .reshape(-1)[0]
                    .item()
                ),
            }
        )

    torch.cuda.synchronize(device)
    export_started = time.perf_counter()
    scene, pending_asset = build_textured_scene(
        np.stack(world_points_list, axis=0),
        np.stack(images_list, axis=0),
        np.stack(masks_list, axis=0),
    )
    scene_path = output_dir / "scene.glb"
    scene.export(scene_path)
    asset = inspect_textured_glb(scene_path, expected_views=len(images))
    if (
        asset["texture_width"],
        asset["texture_height"],
    ) != (
        pending_asset["texture_width"],
        pending_asset["texture_height"],
    ):
        raise ValueError("exported JPEG dimensions differ from the model images")
    torch.cuda.synchronize(device)
    export_seconds = time.perf_counter() - export_started
    total_seconds = time.perf_counter() - total_started

    capability = torch.cuda.get_device_capability(device)
    return {
        "environment": {
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
        },
        "timings": {
            "load_seconds": load_seconds,
            "inference_seconds": inference_seconds,
            "export_seconds": export_seconds,
            "total_seconds": total_seconds,
        },
        "asset": asset,
        "views": view_reports,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a MapAnything multi-image GLB reconstruction"
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = base_report([])
    try:
        images = discover_images(args.input_dir)
        report = base_report(images)
        write_report(args.output_dir, report)

        details = _run_inference(images, args.output_dir)
        json.dumps(details, allow_nan=False)
        report.update(details)
        report["status"] = "success"
        write_report(args.output_dir, report)
        return 0
    except Exception as error:
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
