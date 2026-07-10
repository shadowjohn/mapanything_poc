import io
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image, ImageOps

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

    def _write_valid_input_images(self, count=3):
        paths = []
        for index in range(count):
            path = self.input_dir / f"IMG_{index}.png"
            Image.new("RGB", (8, 8), (index * 30, 20, 10)).save(path)
            paths.append(path)
        return paths

    def _prediction(self, index, mask=None):
        import torch

        if mask is None:
            mask = np.ones((2, 2), dtype=bool)
        return {
            "depth_z": torch.full((1, 2, 2, 1), float(index)),
            "intrinsics": torch.eye(3).unsqueeze(0),
            "camera_poses": torch.eye(4).unsqueeze(0),
            "mask": torch.as_tensor(mask)[None, :, :, None],
            "conf": torch.full((1, 2, 2), float(index + 1)),
            "img_no_norm": torch.full((1, 2, 2, 3), float(index)),
            "metric_scaling_factor": torch.tensor([1.0]),
        }

    def _fill_completed_report(self, images, report):
        count = len(images)
        report["quality_filter"]["usable_view_count"] = count
        report["quality_filter"]["export_view_count"] = count
        if count < 6:
            report["quality_filter"]["warnings"].append(
                "fewer_than_six_export_views"
            )
        report["views"] = [
            {
                "filename": path.name,
                "camera_to_world": [[1.0]],
                "usable_for_export": True,
                "exported": True,
                "exclusion_reason": None,
            }
            for path in images
        ]
        report["asset"] = {
            "texture_mode": "embedded_jpeg_per_view",
            "mesh_count": count,
            "material_count": count,
            "texture_count": count,
            "image_count": count,
            "texture_width": 2,
            "texture_height": 2,
            "glb_bytes": 100,
        }
        report["environment"] = {"python": "test"}
        report["timings"].update(
            {
                "cold_load_seconds": 0.0,
                "image_load_seconds": 0.0,
                "inference_seconds": 0.0,
                "export_seconds": 0.0,
            }
        )
        run.finalize_timings(report["timings"])

    def _stubbed_inference(
        self,
        outputs,
        *,
        mode="balanced",
        confidence_percentile=10.0,
        max_output_views=10,
        error_regex=None,
    ):
        import sys
        import types

        import torch

        paths = [Path(f"IMG_{index}.png") for index in range(len(outputs))]
        report = run.base_report(paths)
        report["quality_filter"] = {
            "mode": mode,
            "blur_threshold": 1.5 if mode == "balanced" else None,
            "duplicate_hamming_threshold": 4 if mode == "balanced" else None,
            "confidence_percentile": (
                confidence_percentile if mode == "balanced" else None
            ),
            "max_output_views": max_output_views if mode == "balanced" else None,
            "input_count": len(paths),
            "inference_count": len(paths),
            "usable_view_count": None,
            "export_view_count": None,
            "warnings": [],
            "images": [
                {
                    "filename": path.name,
                    "sharpness": 2.0 if mode == "balanced" else None,
                    "accepted_for_inference": True,
                    "rejection_reason": None,
                    "near_duplicate_of": None,
                }
                for path in paths
            ],
        }
        report["timings"]["quality_filter_seconds"] = 0.0
        infer_calls = []
        build_calls = []

        class FakeModel:
            def to(self, _device):
                return self

            def eval(self):
                return None

            def infer(self, *_args, **_kwargs):
                raise AssertionError("_run_inference bypassed infer_views")

        class FakeMapAnything:
            @classmethod
            def from_pretrained(cls, _checkpoint):
                return FakeModel()

        models = types.ModuleType("mapanything.models")
        models.MapAnything = FakeMapAnything
        geometry = types.ModuleType("mapanything.utils.geometry")

        def depthmap_to_world_frame(depth, _intrinsics, _camera_to_world):
            self.assertEqual(clock.call_count, 7)
            value = float(depth[0, 0].item())
            return (
                torch.full((2, 2, 3), value),
                torch.ones((2, 2), dtype=torch.bool),
            )

        geometry.depthmap_to_world_frame = depthmap_to_world_frame
        image_utils = types.ModuleType("mapanything.utils.image")
        image_utils.load_images = lambda names: list(names)
        viz = types.ModuleType("mapanything.utils.hf_utils.viz")
        viz.image_mesh = object()

        def infer_views(_model, _views, **options):
            infer_calls.append(options)
            return outputs

        def build_textured_scene(world_points, images, masks):
            self.assertEqual(clock.call_count, 7)
            build_calls.append((world_points, images, masks))
            return object(), {"texture_width": 2, "texture_height": 2}

        def export_scene_atomic(_scene, _pending_asset, _output_dir, expected_views):
            self.assertEqual(clock.call_count, 7)
            return {
                "texture_mode": "embedded_jpeg_per_view",
                "mesh_count": expected_views,
                "material_count": expected_views,
                "texture_count": expected_views,
                "image_count": expected_views,
                "texture_width": 2,
                "texture_height": 2,
                "glb_bytes": 100,
            }

        clock = mock.Mock(side_effect=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 20.0))
        with (
            mock.patch.dict(
                sys.modules,
                {
                    "mapanything.models": models,
                    "mapanything.utils.geometry": geometry,
                    "mapanything.utils.hf_utils.viz": viz,
                    "mapanything.utils.image": image_utils,
                },
            ),
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "reset_peak_memory_stats"),
            mock.patch.object(torch.cuda, "synchronize"),
            mock.patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
            mock.patch.object(torch.cuda, "get_device_name", return_value="test gpu"),
            mock.patch.object(torch.cuda, "max_memory_allocated", return_value=0),
            mock.patch.object(torch.cuda, "max_memory_reserved", return_value=0),
            mock.patch.object(run, "infer_views", side_effect=infer_views),
            mock.patch.object(
                run, "build_textured_scene", side_effect=build_textured_scene
            ),
            mock.patch.object(
                run, "export_scene_atomic", side_effect=export_scene_atomic
            ) as export,
            mock.patch.object(run.time, "perf_counter", clock),
        ):
            call = lambda: run._run_inference(
                paths,
                self.output_dir,
                report,
                mode=mode,
                confidence_percentile=confidence_percentile,
                max_output_views=max_output_views,
            )
            if error_regex is None:
                call()
            else:
                with self.assertRaisesRegex(ValueError, error_regex):
                    call()
        return report, infer_calls, build_calls, export

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

    def test_quality_cli_defaults_ranges_and_base_report_v2(self):
        args = run._parser().parse_args(
            ["--input-dir", str(self.input_dir), "--output-dir", str(self.output_dir)]
        )
        self.assertEqual(args.quality_filter, "balanced")
        self.assertEqual(args.blur_threshold, 1.5)
        self.assertEqual(args.duplicate_hamming_threshold, 4)
        self.assertEqual(args.confidence_percentile, 10.0)
        self.assertEqual(args.max_output_views, 10)
        normalized = run.validate_options(
            mode="balanced",
            blur_threshold=0.0,
            duplicate_hamming_threshold=16,
            confidence_percentile=50.0,
            max_output_views=6,
        )
        self.assertEqual(normalized, (0.0, 16, 50.0, 6))
        invalid = [
            {"blur_threshold": -0.01},
            {"duplicate_hamming_threshold": 17},
            {"confidence_percentile": 50.01},
            {"max_output_views": 11},
            {"confidence_percentile": "not-a-number"},
            {"duplicate_hamming_threshold": "4.5"},
        ]
        defaults = {
            "mode": "balanced",
            "blur_threshold": 1.5,
            "duplicate_hamming_threshold": 4,
            "confidence_percentile": 10.0,
            "max_output_views": 10,
        }
        for override in invalid:
            with self.subTest(override=override), self.assertRaises(ValueError):
                run.validate_options(**(defaults | override))

        report = run.base_report([Path("一.jpg"), Path("二.jpg"), Path("三.jpg")])
        self.assertEqual(report["schema_version"], 2)
        self.assertEqual(report["quality_filter"], {})

    def test_finalize_timings_uses_every_per_request_stage(self):
        timings = {
            "quality_filter_seconds": 1.0,
            "cold_load_seconds": 5.0,
            "image_load_seconds": 2.0,
            "inference_seconds": 3.0,
            "export_seconds": 4.0,
        }
        run.finalize_timings(timings)
        self.assertEqual(timings["load_seconds"], 7.0)
        self.assertEqual(timings["warm_preview_seconds"], 10.0)
        self.assertEqual(timings["total_seconds"], 15.0)
        for invalid in (float("nan"), float("inf"), -0.1):
            broken = timings | {"image_load_seconds": invalid}
            with self.assertRaisesRegex(ValueError, "finite non-negative"):
                run.finalize_timings(broken)

    def test_write_report_rejects_nonfinite_without_overwriting_checkpoint(self):
        report = run.base_report([Path("一.jpg"), Path("二.jpg"), Path("三.jpg")])
        run.write_report(self.output_dir, report)
        original = (self.output_dir / "report.json").read_bytes()
        report["timings"]["broken"] = float("nan")
        with self.assertRaises(ValueError):
            run.write_report(self.output_dir, report)
        self.assertEqual((self.output_dir / "report.json").read_bytes(), original)
        self.assertFalse((self.output_dir / "report.json.tmp").exists())

    def test_validate_completed_report_enforces_count_and_reason_invariants(self):
        report = run.base_report([Path("A.jpg"), Path("B.jpg"), Path("C.jpg")])
        report["quality_filter"] = {
            "mode": "balanced",
            "blur_threshold": 1.5,
            "duplicate_hamming_threshold": 4,
            "confidence_percentile": 10.0,
            "max_output_views": 10,
            "input_count": 3,
            "inference_count": 3,
            "usable_view_count": 3,
            "export_view_count": 3,
            "warnings": ["fewer_than_six_export_views"],
            "images": [
                {
                    "filename": name,
                    "sharpness": 2.0,
                    "accepted_for_inference": True,
                    "rejection_reason": None,
                    "near_duplicate_of": None,
                }
                for name in ("A.jpg", "B.jpg", "C.jpg")
            ],
        }
        report["views"] = [
            {
                "filename": name,
                "usable_for_export": True,
                "exported": True,
                "exclusion_reason": None,
            }
            for name in ("A.jpg", "B.jpg", "C.jpg")
        ]
        report["asset"] = {
            "mesh_count": 3,
            "material_count": 3,
            "texture_count": 3,
            "image_count": 3,
        }
        report["environment"] = {"python": "test"}
        report["timings"] = {
            "quality_filter_seconds": 1.0,
            "cold_load_seconds": 5.0,
            "image_load_seconds": 2.0,
            "inference_seconds": 3.0,
            "export_seconds": 4.0,
        }
        run.finalize_timings(report["timings"])
        run.validate_completed_report(report)

        broken = json.loads(json.dumps(report))
        broken["asset"]["image_count"] = 2
        with self.assertRaisesRegex(ValueError, "asset counts"):
            run.validate_completed_report(broken)

        broken = json.loads(json.dumps(report))
        broken["timings"]["warm_preview_seconds"] = 999
        with self.subTest(invariant="timing formulas"), self.assertRaisesRegex(
            ValueError, "timing formulas"
        ):
            run.validate_completed_report(broken)

        for accepted, reason in ((True, "blurry"), (False, None)):
            broken = json.loads(json.dumps(report))
            broken["quality_filter"]["images"][0][
                "accepted_for_inference"
            ] = accepted
            broken["quality_filter"]["images"][0]["rejection_reason"] = reason
            with self.subTest(accepted=accepted, reason=reason), self.assertRaisesRegex(
                ValueError, "accepted flag and rejection reason"
            ):
                run.validate_completed_report(broken)

        for mode in (None, "invalid"):
            broken = json.loads(json.dumps(report))
            if mode is None:
                del broken["quality_filter"]["mode"]
            else:
                broken["quality_filter"]["mode"] = mode
            with self.subTest(mode=mode), self.assertRaisesRegex(
                ValueError, "quality filter mode"
            ):
                run.validate_completed_report(broken)

    def test_prepare_output_removes_only_stale_generated_files(self):
        self.output_dir.mkdir()
        for name in ("scene.glb", "scene.glb.tmp", "report.json.tmp"):
            (self.output_dir / name).write_bytes(b"stale")
        (self.output_dir / "report.json").write_text("{}", encoding="utf-8")
        run.prepare_output(self.output_dir)
        self.assertTrue((self.output_dir / "report.json").exists())
        self.assertFalse((self.output_dir / "scene.glb").exists())
        self.assertFalse((self.output_dir / "scene.glb.tmp").exists())
        self.assertFalse((self.output_dir / "report.json.tmp").exists())

    def test_export_scene_atomic_validates_before_replace(self):
        calls = []

        class FakeScene:
            def export(self, path, *, file_type):
                calls.append((Path(path).name, file_type))
                Path(path).write_bytes(b"temporary glb")

        pending = {"texture_width": 4, "texture_height": 4}
        asset = pending | {
            "mesh_count": 1,
            "material_count": 1,
            "texture_count": 1,
            "image_count": 1,
        }
        with mock.patch.object(run, "inspect_textured_glb", return_value=asset):
            actual = run.export_scene_atomic(FakeScene(), pending, self.output_dir, 1)
        self.assertEqual(actual, asset)
        self.assertEqual(calls, [("scene.glb.tmp", "glb")])
        self.assertTrue((self.output_dir / "scene.glb").exists())
        self.assertFalse((self.output_dir / "scene.glb.tmp").exists())

    def test_export_scene_atomic_failure_leaves_no_scene(self):
        class FakeScene:
            def export(self, path, *, file_type):
                Path(path).write_bytes(b"invalid")

        with mock.patch.object(
            run, "inspect_textured_glb", side_effect=ValueError("invalid GLB")
        ):
            with self.assertRaisesRegex(ValueError, "invalid GLB"):
                run.export_scene_atomic(
                    FakeScene(),
                    {"texture_width": 4, "texture_height": 4},
                    self.output_dir,
                    1,
                )
        self.assertFalse((self.output_dir / "scene.glb").exists())
        self.assertFalse((self.output_dir / "scene.glb.tmp").exists())

    def test_main_malformed_numeric_option_writes_failed_report(self):
        with mock.patch("sys.stderr", new_callable=io.StringIO):
            return_code = run.main(
                [
                    "--input-dir",
                    str(self.input_dir),
                    "--output-dir",
                    str(self.output_dir),
                    "--confidence-percentile",
                    "not-a-number",
                ]
            )

        self.assertEqual(return_code, 1)
        report = json.loads((self.output_dir / "report.json").read_text("utf-8"))
        self.assertEqual(report["schema_version"], 2)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error"]["class"], "ValueError")
        self.assertIn("numeric quality options", report["error"]["message"])

    def test_main_persists_quality_shortfall_before_inference(self):
        paths = self._write_valid_input_images()
        self.output_dir.mkdir()
        (self.output_dir / "scene.glb").write_bytes(b"stale")
        decisions = [
            {
                "filename": path.name,
                "sharpness": 2.0,
                "accepted_for_inference": index < 2,
                "rejection_reason": None if index < 2 else "blurry",
                "near_duplicate_of": None,
            }
            for index, path in enumerate(paths)
        ]
        with (
            mock.patch.object(
                run, "filter_input_images", return_value=(paths[:2], decisions)
            ),
            mock.patch.object(run, "_run_inference") as inference,
            mock.patch("sys.stderr", new_callable=io.StringIO),
        ):
            exit_code = run.main(
                ["--input-dir", str(self.input_dir), "--output-dir", str(self.output_dir)]
            )
        self.assertEqual(exit_code, 1)
        inference.assert_not_called()
        report = json.loads((self.output_dir / "report.json").read_text("utf-8"))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["quality_filter"]["inference_count"], 2)
        self.assertEqual(report["quality_filter"]["images"], decisions)
        self.assertFalse((self.output_dir / "scene.glb").exists())

    def test_main_preserves_inference_checkpoint_on_failure(self):
        paths = self._write_valid_input_images()
        decisions = [
            {
                "filename": path.name,
                "sharpness": 2.0,
                "accepted_for_inference": True,
                "rejection_reason": None,
                "near_duplicate_of": None,
            }
            for path in paths
        ]

        def fail_after_checkpoint(images, output_dir, report, **options):
            report["views"] = [{"filename": path.name} for path in images]
            report["quality_filter"]["usable_view_count"] = 3
            run.write_report(output_dir, report)
            raise RuntimeError("postprocess failed")

        with (
            mock.patch.object(
                run, "filter_input_images", return_value=(paths, decisions)
            ),
            mock.patch.object(
                run, "_run_inference", side_effect=fail_after_checkpoint
            ),
            mock.patch("sys.stderr", new_callable=io.StringIO),
        ):
            exit_code = run.main(
                ["--input-dir", str(self.input_dir), "--output-dir", str(self.output_dir)]
            )
        self.assertEqual(exit_code, 1)
        report = json.loads((self.output_dir / "report.json").read_text("utf-8"))
        self.assertEqual(report["error"]["message"], "postprocess failed")
        self.assertEqual(len(report["views"]), 3)
        self.assertEqual(report["quality_filter"]["usable_view_count"], 3)

    def test_main_nested_nan_falls_back_to_last_finite_checkpoint(self):
        paths = self._write_valid_input_images()
        decisions = [
            {
                "filename": path.name,
                "sharpness": 2.0,
                "accepted_for_inference": True,
                "rejection_reason": None,
                "near_duplicate_of": None,
            }
            for path in paths
        ]

        def add_nan_after_checkpoint(images, output_dir, report, **_options):
            self._fill_completed_report(images, report)
            report["checkpoint_data"] = {"stage": "post_inference", "kept": [1, 2]}
            run.write_report(output_dir, report)
            report["views"][0]["camera_to_world"] = [[float("nan")]]
            (output_dir / "scene.glb").write_bytes(b"published")
            (output_dir / "scene.glb.tmp").write_bytes(b"temporary")

        with (
            mock.patch.object(
                run, "filter_input_images", return_value=(paths, decisions)
            ),
            mock.patch.object(
                run, "_run_inference", side_effect=add_nan_after_checkpoint
            ),
            mock.patch("sys.stderr", new_callable=io.StringIO),
        ):
            exit_code = run.main(
                ["--input-dir", str(self.input_dir), "--output-dir", str(self.output_dir)]
            )

        self.assertEqual(exit_code, 1)
        report = json.loads((self.output_dir / "report.json").read_text("utf-8"))
        json.dumps(report, allow_nan=False)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error"]["class"], "ValueError")
        self.assertIn("Out of range float values", report["error"]["message"])
        self.assertEqual(
            report["checkpoint_data"], {"stage": "post_inference", "kept": [1, 2]}
        )
        self.assertEqual(len(report["views"]), 3)
        self.assertNotIn("asset", report)
        self.assertFalse((self.output_dir / "scene.glb").exists())
        self.assertFalse((self.output_dir / "scene.glb.tmp").exists())

    def test_main_nested_nan_uses_safe_base_when_checkpoint_is_unreadable(self):
        paths = self._write_valid_input_images()
        decisions = [
            {
                "filename": path.name,
                "sharpness": 2.0,
                "accepted_for_inference": True,
                "rejection_reason": None,
                "near_duplicate_of": None,
            }
            for path in paths
        ]

        def corrupt_checkpoint(images, output_dir, report, **_options):
            self._fill_completed_report(images, report)
            report["views"][0]["camera_to_world"] = [[float("nan")]]
            (output_dir / "report.json").write_text("{broken", encoding="utf-8")

        with (
            mock.patch.object(
                run, "filter_input_images", return_value=(paths, decisions)
            ),
            mock.patch.object(run, "_run_inference", side_effect=corrupt_checkpoint),
            mock.patch("sys.stderr", new_callable=io.StringIO),
        ):
            exit_code = run.main(
                ["--input-dir", str(self.input_dir), "--output-dir", str(self.output_dir)]
            )

        self.assertEqual(exit_code, 1)
        try:
            report = json.loads((self.output_dir / "report.json").read_text("utf-8"))
        except json.JSONDecodeError as error:
            self.fail(f"failed report is not valid JSON: {error}")
        json.dumps(report, allow_nan=False)
        self.assertEqual(report["schema_version"], 2)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error"]["class"], "ValueError")
        self.assertIn("Out of range float values", report["error"]["message"])

    def test_main_off_records_null_thresholds_and_all_inputs(self):
        paths = self._write_valid_input_images()
        decisions = [
            {
                "filename": path.name,
                "sharpness": None,
                "accepted_for_inference": True,
                "rejection_reason": None,
                "near_duplicate_of": None,
            }
            for path in paths
        ]

        def finish_without_gpu(images, output_dir, report, **options):
            self.assertEqual(options["mode"], "off")
            report["quality_filter"]["usable_view_count"] = len(images)
            report["quality_filter"]["export_view_count"] = len(images)
            report["quality_filter"]["warnings"].append(
                "fewer_than_six_export_views"
            )
            report["views"] = [
                {
                    "filename": path.name,
                    "usable_for_export": True,
                    "exported": True,
                    "exclusion_reason": None,
                }
                for path in images
            ]
            report["asset"] = {
                "mesh_count": len(images),
                "material_count": len(images),
                "texture_count": len(images),
                "image_count": len(images),
            }
            report["environment"] = {"python": "test"}
            report["timings"].update(
                {
                    "cold_load_seconds": 0.0,
                    "image_load_seconds": 0.0,
                    "inference_seconds": 0.0,
                    "export_seconds": 0.0,
                }
            )
            run.finalize_timings(report["timings"])

        with (
            mock.patch.object(
                run, "filter_input_images", return_value=(paths, decisions)
            ),
            mock.patch.object(run, "_run_inference", side_effect=finish_without_gpu),
        ):
            exit_code = run.main(
                [
                    "--input-dir",
                    str(self.input_dir),
                    "--output-dir",
                    str(self.output_dir),
                    "--quality-filter",
                    "off",
                ]
            )
        self.assertEqual(exit_code, 0)
        report = json.loads((self.output_dir / "report.json").read_text("utf-8"))
        quality = report["quality_filter"]
        self.assertEqual(report["status"], "success")
        self.assertIsNone(quality["blur_threshold"])
        self.assertIsNone(quality["duplicate_hamming_threshold"])
        self.assertIsNone(quality["confidence_percentile"])
        self.assertIsNone(quality["max_output_views"])
        self.assertEqual(quality["inference_count"], 3)

    def test_run_inference_uses_infer_views_and_selected_reconstructions(self):
        outputs = [
            self._prediction(index, np.zeros((2, 2), dtype=bool) if index == 1 else None)
            for index in range(4)
        ]
        report, infer_calls, build_calls, export = self._stubbed_inference(
            outputs, mode="balanced", confidence_percentile=17.0
        )

        self.assertEqual(
            infer_calls, [{"mode": "balanced", "confidence_percentile": 17.0}]
        )
        self.assertEqual(len(build_calls), 1)
        np.testing.assert_array_equal(
            build_calls[0][0][:, 0, 0, 0], np.asarray([0.0, 2.0, 3.0])
        )
        self.assertEqual(
            [view["exported"] for view in report["views"]],
            [True, False, True, True],
        )
        export.assert_called_once()
        self.assertEqual(report["timings"]["export_seconds"], 14.0)

    def test_run_inference_off_passes_options_and_keeps_all_usable_views(self):
        valid_quad = np.ones((2, 2), dtype=bool)
        outputs = [self._prediction(index, valid_quad) for index in range(3)]
        report, infer_calls, build_calls, export = self._stubbed_inference(
            outputs, mode="off"
        )

        self.assertEqual(
            infer_calls, [{"mode": "off", "confidence_percentile": 10.0}]
        )
        self.assertEqual(build_calls[0][0].shape[0], 3)
        self.assertTrue(all(view["exported"] for view in report["views"]))
        export.assert_called_once()

    def test_run_inference_selects_six_floor_bin_views_from_eleven(self):
        outputs = [self._prediction(index) for index in range(11)]
        report, _infer_calls, build_calls, export = self._stubbed_inference(
            outputs, mode="balanced", max_output_views=6
        )

        selected = [0, 2, 4, 6, 8, 10]
        self.assertEqual(
            [index for index, view in enumerate(report["views"]) if view["exported"]],
            selected,
        )
        self.assertEqual(
            [view["filename"] for view in report["views"] if view["exported"]],
            [f"IMG_{index}.png" for index in selected],
        )
        np.testing.assert_array_equal(
            build_calls[0][0][:, 0, 0, 0], np.asarray(selected, dtype=np.float32)
        )
        np.testing.assert_array_equal(
            build_calls[0][1][:, 0, 0, 0], np.asarray(selected, dtype=np.float32)
        )
        self.assertEqual(report["quality_filter"]["export_view_count"], 6)
        self.assertEqual(report["asset"]["mesh_count"], 6)
        self.assertEqual(report["asset"]["image_count"], 6)
        self.assertEqual(export.call_args.kwargs["expected_views"], 6)

    def test_run_inference_usable_shortfall_does_not_mark_exports(self):
        outputs = [
            self._prediction(index, np.zeros((2, 2), dtype=bool) if index == 2 else None)
            for index in range(3)
        ]
        report, _infer_calls, build_calls, export = self._stubbed_inference(
            outputs, error_regex="at least 3"
        )

        self.assertEqual(report["quality_filter"]["usable_view_count"], 2)
        self.assertEqual(report["quality_filter"]["export_view_count"], 0)
        self.assertFalse(any(view["exported"] for view in report["views"]))
        self.assertEqual(build_calls, [])
        export.assert_not_called()

    def test_main_validates_completed_report_before_success(self):
        paths = self._write_valid_input_images()
        decisions = [
            {
                "filename": path.name,
                "sharpness": 2.0,
                "accepted_for_inference": True,
                "rejection_reason": None,
                "near_duplicate_of": None,
            }
            for path in paths
        ]

        def invalid_inference(_images, output_dir, _report, **_options):
            (output_dir / "scene.glb").write_bytes(b"published")
            (output_dir / "scene.glb.tmp").write_bytes(b"temporary")

        with (
            mock.patch.object(
                run, "filter_input_images", return_value=(paths, decisions)
            ),
            mock.patch.object(run, "_run_inference", side_effect=invalid_inference),
            mock.patch.object(
                run,
                "validate_completed_report",
                side_effect=ValueError("asset counts do not match exported views"),
            ) as validate,
            mock.patch("sys.stderr", new_callable=io.StringIO),
        ):
            return_code = run.main(
                [
                    "--input-dir",
                    str(self.input_dir),
                    "--output-dir",
                    str(self.output_dir),
                ]
            )

        self.assertEqual(return_code, 1)
        validate.assert_called_once()
        report = json.loads((self.output_dir / "report.json").read_text("utf-8"))
        self.assertEqual(report["status"], "failed")
        self.assertIn("asset counts", report["error"]["message"])
        self.assertFalse((self.output_dir / "scene.glb").exists())
        self.assertFalse((self.output_dir / "scene.glb.tmp").exists())

    def test_sharpness_and_dhash_are_deterministic(self):
        flat = np.zeros((5, 5), dtype=np.float32)
        checker = (np.indices((7, 7)).sum(axis=0) % 2).astype(np.float32)
        self.assertEqual(run.sharpness_score(flat), 0.0)
        self.assertGreater(run.sharpness_score(checker), 1.5)

        ascending = Image.fromarray(
            np.tile(np.arange(9, dtype=np.uint8), (8, 1)), mode="L"
        )
        descending = Image.fromarray(
            np.tile(np.arange(8, -1, -1, dtype=np.uint8), (8, 1)), mode="L"
        )
        self.assertEqual(run.dhash64(ascending), (1 << 64) - 1)
        self.assertEqual(run.dhash64(descending), 0)

    def test_analyze_image_quality_applies_exif_orientation(self):
        pixels = np.arange(72, dtype=np.uint8).reshape(8, 9)
        tagged = self.input_dir / "tagged.png"
        manual = self.input_dir / "manual.png"
        exif = Image.Exif()
        exif[274] = 6
        Image.fromarray(pixels, mode="L").save(tagged, exif=exif)
        with Image.open(tagged) as opened:
            ImageOps.exif_transpose(opened).save(manual)

        tagged_score, tagged_hash = run.analyze_image_quality(tagged, enabled=True)
        manual_score, manual_hash = run.analyze_image_quality(manual, enabled=True)
        self.assertAlmostEqual(tagged_score, manual_score, places=7)
        self.assertEqual(tagged_hash, manual_hash)

    def test_analyze_image_quality_rejects_tiny_image_with_filename(self):
        path = self.input_dir / "tiny.png"
        Image.new("L", (2, 5)).save(path)
        with self.assertRaisesRegex(ValueError, r"tiny\.png.*image_too_small"):
            run.analyze_image_quality(path, enabled=True)

    def test_analyze_image_quality_off_still_rejects_corrupt_image(self):
        path = self.input_dir / "corrupt.jpg"
        path.write_bytes(b"not an image")
        with self.assertRaisesRegex(ValueError, r"corrupt\.jpg.*cannot decode"):
            run.analyze_image_quality(path, enabled=False)

    def test_filter_input_images_keeps_fixed_anchor_and_best_winner(self):
        paths = [self.input_dir / f"{name}.jpg" for name in "ABCDE"]
        metrics = [
            (2.0, 0b00000),
            (3.0, 0b00001),
            (2.5, 0b11111),
            (2.2, 0xFF00),
            (2.1, 0xFF0000),
        ]
        with mock.patch.object(run, "analyze_image_quality", side_effect=metrics):
            accepted, decisions = run.filter_input_images(
                paths,
                mode="balanced",
                blur_threshold=1.5,
                duplicate_hamming_threshold=4,
            )

        self.assertEqual(accepted, paths[1:])
        self.assertEqual([row["filename"] for row in decisions], [p.name for p in paths])
        self.assertEqual(
            decisions[0],
            {
                "filename": "A.jpg",
                "sharpness": 2.0,
                "accepted_for_inference": False,
                "rejection_reason": "near_duplicate",
                "near_duplicate_of": "B.jpg",
            },
        )
        for row in decisions[1:]:
            self.assertTrue(row["accepted_for_inference"])
            self.assertIsNone(row["rejection_reason"])
            self.assertIsNone(row["near_duplicate_of"])

    def test_filter_input_images_keeps_equal_blur_threshold_and_rejects_below(self):
        paths = [self.input_dir / f"{index}.jpg" for index in range(5)]
        metrics = [
            (1.5, 0x00000000),
            (1.49, 0x000000FF),
            (2.0, 0x0000FF00),
            (2.1, 0x00FF0000),
            (2.2, 0xFF000000),
        ]
        with mock.patch.object(run, "analyze_image_quality", side_effect=metrics):
            accepted, decisions = run.filter_input_images(
                paths,
                mode="balanced",
                blur_threshold=1.5,
                duplicate_hamming_threshold=4,
            )
        self.assertIn(paths[0], accepted)
        self.assertNotIn(paths[1], accepted)
        self.assertEqual(decisions[1]["rejection_reason"], "blurry")

    def test_filter_input_images_off_only_validates_decode(self):
        paths = [self.input_dir / f"{index}.jpg" for index in range(3)]
        with mock.patch.object(
            run, "analyze_image_quality", return_value=(None, None)
        ) as analyze:
            accepted, decisions = run.filter_input_images(
                paths,
                mode="off",
                blur_threshold=1.5,
                duplicate_hamming_threshold=4,
            )
        self.assertEqual(accepted, paths)
        self.assertEqual(analyze.call_count, 3)
        self.assertTrue(
            all(call.kwargs == {"enabled": False} for call in analyze.call_args_list)
        )
        self.assertTrue(all(row["sharpness"] is None for row in decisions))

    def test_filter_input_images_uses_earliest_anchor_and_sharpness_tie(self):
        paths = [self.input_dir / f"{name}.jpg" for name in "ABCDE"]
        metrics = [
            (2.0, 0x00),
            (2.0, 0xFF),
            (3.0, 0x0F),
            (2.0, 0xFF00),
            (2.0, 0xFF0000),
        ]
        with mock.patch.object(run, "analyze_image_quality", side_effect=metrics):
            accepted, decisions = run.filter_input_images(
                paths,
                mode="balanced",
                blur_threshold=1.5,
                duplicate_hamming_threshold=4,
            )
        self.assertEqual(accepted, paths[1:])
        self.assertEqual(decisions[0]["near_duplicate_of"], "C.jpg")

        tie_metrics = [(2.0, 0x00), (2.0, 0x01), (2.0, 0xFF00)]
        with mock.patch.object(run, "analyze_image_quality", side_effect=tie_metrics):
            tie_accepted, tie_decisions = run.filter_input_images(
                paths[:3],
                mode="balanced",
                blur_threshold=1.5,
                duplicate_hamming_threshold=4,
            )
        self.assertEqual(tie_accepted, [paths[0], paths[2]])
        self.assertEqual(tie_decisions[1]["near_duplicate_of"], "A.jpg")

    def test_infer_views_enables_only_builtin_learned_confidence(self):
        class FakeModel:
            def __init__(self):
                self.calls = []

            def infer(self, views, **kwargs):
                self.calls.append((views, kwargs))
                return ["output"]

        model = FakeModel()
        self.assertEqual(
            run.infer_views(
                model,
                ["view"],
                mode="balanced",
                confidence_percentile=10.0,
            ),
            ["output"],
        )
        balanced = model.calls[-1][1]
        self.assertTrue(balanced["apply_confidence_mask"])
        self.assertEqual(balanced["confidence_percentile"], 10.0)
        self.assertFalse(balanced["use_multiview_confidence"])

        run.infer_views(model, ["view"], mode="off", confidence_percentile=10.0)
        off = model.calls[-1][1]
        self.assertFalse(off["apply_confidence_mask"])
        self.assertNotIn("confidence_percentile", off)
        self.assertFalse(off["use_multiview_confidence"])

    def test_summarize_view_handles_empty_no_quad_and_usable_masks(self):
        confidence = np.full((3, 3), 2.0, dtype=np.float32)

        empty = run.summarize_view(
            np.zeros((3, 3), dtype=bool), confidence, require_quad=True
        )
        self.assertEqual(empty["valid_fraction"], 0.0)
        self.assertIsNone(empty["mean_valid_confidence"])
        self.assertIsNone(empty["selection_score"])
        self.assertFalse(empty["usable_for_export"])
        self.assertEqual(empty["exclusion_reason"], "no_valid_pixels_after_confidence")

        diagonal_mask = np.eye(3, dtype=bool)
        no_quad = run.summarize_view(diagonal_mask, confidence, require_quad=True)
        self.assertAlmostEqual(no_quad["mean_valid_confidence"], 2.0)
        self.assertIsNone(no_quad["selection_score"])
        self.assertFalse(no_quad["usable_for_export"])
        self.assertEqual(no_quad["exclusion_reason"], "no_valid_quad_after_confidence")

        usable_mask = np.zeros((3, 3), dtype=bool)
        usable_mask[:2, :2] = True
        usable = run.summarize_view(usable_mask, confidence, require_quad=True)
        self.assertAlmostEqual(usable["valid_fraction"], 4 / 9)
        self.assertAlmostEqual(usable["mean_valid_confidence"], 2.0)
        self.assertAlmostEqual(usable["selection_score"], 8 / 9)
        self.assertTrue(usable["usable_for_export"])
        self.assertFalse(usable["exported"])
        self.assertEqual(usable["exclusion_reason"], "not_selected_as_representative")

        legacy = run.summarize_view(diagonal_mask, confidence, require_quad=False)
        self.assertTrue(legacy["usable_for_export"])

    def test_select_representative_indices_uses_floor_bins_and_earliest_ties(self):
        self.assertEqual(run.select_representative_indices([9, 1, 2], 10), [0, 1, 2])
        scores = [0, 1, 9, 7, 2, 6, 8, 10, 3, 4, 5]
        self.assertEqual(
            run.select_representative_indices(scores, 6),
            [0, 2, 3, 6, 7, 10],
        )
        self.assertEqual(
            run.select_representative_indices([1.0] * 12, 6),
            [0, 2, 4, 6, 8, 10],
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            run.select_representative_indices([1.0, float("nan")], 6)
        with self.assertRaisesRegex(ValueError, "6 through 10"):
            run.select_representative_indices([1.0] * 12, 11)

    def test_select_export_view_indices_maps_usable_reports_without_early_export(self):
        reports = [
            {
                "selection_score": score,
                "usable_for_export": usable,
                "exported": False,
                "exclusion_reason": (
                    "not_selected_as_representative"
                    if usable
                    else "no_valid_quad_after_confidence"
                ),
            }
            for score, usable in ((9.0, True), (0.0, False), (3.0, True), (4.0, True))
        ]
        selected = run.select_export_view_indices(
            reports, mode="balanced", max_output_views=10
        )
        self.assertEqual(selected, [0, 2, 3])
        self.assertEqual(
            [index for index, report in enumerate(reports) if report["exported"]],
            selected,
        )
        self.assertTrue(
            all(reports[index]["exclusion_reason"] is None for index in selected)
        )

        too_few = [dict(report, exported=False) for report in reports[:2]]
        with self.assertRaisesRegex(ValueError, "at least 3"):
            run.select_export_view_indices(
                too_few, mode="balanced", max_output_views=10
            )
        self.assertFalse(any(report["exported"] for report in too_few))

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
                "image_count": 1,
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
        self.assertEqual(asset["image_count"], 1)
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

    def test_inspector_rejects_any_extra_primitive_attribute(self):
        path = self._export_minimal_textured_glb()
        *_, document, binary = _read_glb(path)
        attributes = document["meshes"][0]["primitives"][0]["attributes"]
        attributes["NORMAL"] = attributes["POSITION"]
        _write_glb(path, document, binary)

        with self.assertRaisesRegex(
            ValueError, "exactly POSITION and TEXCOORD_0"
        ):
            run.inspect_textured_glb(path, expected_views=1)


if __name__ == "__main__":
    unittest.main()
