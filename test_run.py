import io
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from mapanything_poc import run
from mapanything_poc.run import base_report, discover_images, write_report


def _read_glb(path: Path):
    raw = path.read_bytes()
    magic, version, total_length = struct.unpack_from("<4sII", raw)
    json_length, json_type = struct.unpack_from("<II", raw, 12)
    json_start = 20
    json_end = json_start + json_length
    document = json.loads(raw[json_start:json_end].decode("utf-8"))
    binary_length, binary_type = struct.unpack_from("<II", raw, json_end)
    binary_start = json_end + 8
    binary = raw[binary_start : binary_start + binary_length]
    return raw, magic, version, total_length, json_type, binary_type, document, binary


def _write_glb(path: Path, document: dict, binary: bytes):
    json_chunk = json.dumps(document, separators=(",", ":")).encode("utf-8")
    json_chunk += b" " * (-len(json_chunk) % 4)
    binary += b"\x00" * (-len(binary) % 4)
    total_length = 12 + 8 + len(json_chunk) + 8 + len(binary)
    path.write_bytes(
        struct.pack("<4sII", b"glTF", 2, total_length)
        + struct.pack("<II", len(json_chunk), 0x4E4F534A)
        + json_chunk
        + struct.pack("<II", len(binary), 0x004E4942)
        + binary
    )


def _read_vec2_accessor(document: dict, binary: bytes, accessor_index: int):
    accessor = document["accessors"][accessor_index]
    view = document["bufferViews"][accessor["bufferView"]]
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    stride = view.get("byteStride", 8)
    return np.asarray(
        [struct.unpack_from("<2f", binary, start + index * stride) for index in range(accessor["count"])],
        dtype=np.float32,
    )


class RunHelpersTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.input_dir = root / "input"
        self.output_dir = root / "output"
        self.input_dir.mkdir()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _export_minimal_textured_glb(self):
        rows, columns = np.indices((2, 2), dtype=np.float32)
        world_points = np.stack((columns, rows, rows + columns), axis=-1)[None]
        images = np.ones((1, 2, 2, 3), dtype=np.float32)
        masks = np.ones((1, 2, 2), dtype=bool)
        scene, _asset = run.build_textured_scene(world_points, images, masks)
        self.output_dir.mkdir()
        path = self.output_dir / "scene.glb"
        scene.export(path)
        return path

    def test_discovers_supported_images_in_natural_order(self):
        for name in ("IMG_10.jpg", "IMG_2.JPG", "IMG_1.png", "notes.txt"):
            (self.input_dir / name).touch()
        self.assertEqual(
            [p.name for p in discover_images(self.input_dir)],
            ["IMG_1.png", "IMG_2.JPG", "IMG_10.jpg"],
        )

    def test_rejects_fewer_than_three_images(self):
        (self.input_dir / "one.jpg").touch()
        with self.assertRaisesRegex(ValueError, "at least 3"):
            discover_images(self.input_dir)

    def test_writes_base_report_as_utf8_json(self):
        images = [Path("一.jpg"), Path("二.jpg"), Path("三.jpg")]
        report = base_report(images)
        write_report(self.output_dir, report)
        loaded = json.loads((self.output_dir / "report.json").read_text("utf-8"))
        self.assertEqual(loaded["checkpoint_id"], "facebook/map-anything-apache")
        self.assertEqual(loaded["inputs"], ["一.jpg", "二.jpg", "三.jpg"])

    def test_pixel_center_uv_uses_trimesh_lower_left_convention(self):
        actual = run.pixel_center_uv(np.array([0, 3, 8, 11]), height=3, width=4)
        expected = np.array(
            [[0.125, 5 / 6], [0.875, 5 / 6], [0.125, 1 / 6], [0.875, 1 / 6]]
        )
        np.testing.assert_allclose(actual, expected)

    def test_pixel_center_uv_rejects_duplicate_indices(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            run.pixel_center_uv(np.array([0, 1, 1, 2]), height=2, width=2)

    def test_pixel_center_uv_rejects_non_increasing_indices(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            run.pixel_center_uv(np.array([0, 2, 1, 3]), height=2, width=2)

    def test_masked_directional_scene_exports_one_embedded_jpeg_texture(self):
        rows, columns = np.indices((4, 4), dtype=np.float32)
        world_points = np.stack((columns, rows, rows + columns / 10), axis=-1)[None]
        image = np.empty((4, 4, 3), dtype=np.float32)
        image[:2, :2] = (1, 0, 0)
        image[:2, 2:] = (0, 1, 0)
        image[2:, :2] = (0, 0, 1)
        image[2:, 2:] = (1, 1, 0)
        images = image[None]
        mask = np.ones((4, 4), dtype=bool)
        mask[0, 3] = False
        masks = mask[None]

        scene, pending_asset = run.build_textured_scene(world_points, images, masks)
        self.assertEqual(list(scene.geometry), ["view_0000"])
        self.assertEqual(
            pending_asset,
            {
                "texture_mode": "embedded_jpeg_per_view",
                "mesh_count": 1,
                "material_count": 1,
                "texture_count": 1,
                "texture_width": 4,
                "texture_height": 4,
                "glb_bytes": None,
            },
        )
        glb_path = self.output_dir / "scene.glb"
        self.output_dir.mkdir()
        scene.export(glb_path)
        asset = run.inspect_textured_glb(glb_path, expected_views=1)

        self.assertEqual(asset["texture_mode"], "embedded_jpeg_per_view")
        self.assertEqual(asset["mesh_count"], 1)
        self.assertEqual(asset["material_count"], 1)
        self.assertEqual(asset["texture_count"], 1)
        self.assertEqual((asset["texture_width"], asset["texture_height"]), (4, 4))
        self.assertEqual(asset["glb_bytes"], glb_path.stat().st_size)

        raw, magic, version, length, json_type, binary_type, document, binary = _read_glb(glb_path)
        self.assertEqual(magic, b"glTF")
        self.assertEqual(version, 2)
        self.assertEqual(length, len(raw))
        self.assertEqual(json_type, 0x4E4F534A)
        self.assertEqual(binary_type, 0x004E4942)
        self.assertEqual(len(document["meshes"]), 1)
        self.assertEqual(len(document["materials"]), 1)
        self.assertEqual(len(document["textures"]), 1)
        self.assertEqual(len(document["images"]), 1)

        primitive = document["meshes"][0]["primitives"][0]
        self.assertEqual(set(primitive["attributes"]), {"POSITION", "TEXCOORD_0"})
        self.assertNotIn("COLOR_0", primitive["attributes"])
        self.assertEqual(primitive["material"], 0)
        material = document["materials"][0]
        self.assertTrue(material["doubleSided"])
        self.assertEqual(material["pbrMetallicRoughness"]["baseColorTexture"]["index"], 0)
        self.assertEqual(material["pbrMetallicRoughness"]["metallicFactor"], 0.0)
        self.assertEqual(material["pbrMetallicRoughness"]["roughnessFactor"], 1.0)
        self.assertEqual(document["textures"][0]["source"], 0)

        image_entry = document["images"][0]
        self.assertEqual(image_entry["mimeType"], "image/jpeg")
        self.assertIn("bufferView", image_entry)
        self.assertNotIn("uri", image_entry)
        image_view = document["bufferViews"][image_entry["bufferView"]]
        image_start = image_view.get("byteOffset", 0)
        image_bytes = binary[image_start : image_start + image_view["byteLength"]]
        self.assertEqual(image_bytes[:3], b"\xff\xd8\xff")
        with Image.open(io.BytesIO(image_bytes)) as decoded:
            decoded.load()
            self.assertEqual(decoded.format, "JPEG")
            self.assertEqual(decoded.size, (4, 4))
            self.assertGreater(decoded.width, 0)
            self.assertGreater(decoded.height, 0)

        texture_uv = _read_vec2_accessor(
            document, binary, primitive["attributes"]["TEXCOORD_0"]
        )
        retained_indices = np.delete(np.arange(16), 3)
        retained_rows, retained_columns = np.divmod(retained_indices, 4)
        expected_glb_uv = np.column_stack(
            ((retained_columns + 0.5) / 4, (retained_rows + 0.5) / 4)
        )
        np.testing.assert_allclose(texture_uv, expected_glb_uv)

        from mapanything.utils.viz import predictions_to_glb

        official_scene = predictions_to_glb(
            {
                "world_points": world_points.copy(),
                "images": images.copy(),
                "final_masks": masks.copy(),
            },
            as_mesh=True,
        )
        textured_mesh = next(iter(scene.geometry.values()))
        official_mesh = next(iter(official_scene.geometry.values()))
        self.assertEqual(len(textured_mesh.vertices), len(official_mesh.vertices))
        self.assertEqual(len(textured_mesh.faces), len(official_mesh.faces))
        np.testing.assert_allclose(scene.bounds, official_scene.bounds)

    def test_inspector_rejects_declared_buffer_shorter_than_buffer_views(self):
        path = self._export_minimal_textured_glb()
        *_, document, binary = _read_glb(path)
        document["buffers"][0]["byteLength"] = 1
        _write_glb(path, document, binary)

        with self.assertRaisesRegex(ValueError, "buffer"):
            run.inspect_textured_glb(path, expected_views=1)

    def test_inspector_rejects_image_buffer_view_without_buffer(self):
        path = self._export_minimal_textured_glb()
        *_, document, binary = _read_glb(path)
        image_view = document["images"][0]["bufferView"]
        del document["bufferViews"][image_view]["buffer"]
        _write_glb(path, document, binary)

        with self.assertRaisesRegex(ValueError, "bufferView"):
            run.inspect_textured_glb(path, expected_views=1)


if __name__ == "__main__":
    unittest.main()
