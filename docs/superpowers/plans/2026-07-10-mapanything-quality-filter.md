# MapAnything Quality Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在維持單一 textured `scene.glb` 快速預覽的前提下，加入可校正的模糊／近重複篩選、MapAnything learned confidence mask，以及最多 6～10 個時序代表視角輸出。

**Architecture:** 延續單檔 CLI，不新增 production module、class 或 dependency；測試可使用 method-local fake class。先以 Pillow/NumPy 在推論前做品質 gate，再沿用 MapAnything 既有 confidence mask；推論後只以純函式摘要 combined mask 與分箱選圖，最後將 selected arrays 送入既有 textured scene builder。`report.json` 以同一個 mutable dict 分階段原子寫入，GLB 先驗證暫存檔再發布。

**Tech Stack:** Windows、Python 3.12、stdlib `unittest`/`argparse`/`json`/`pathlib`、NumPy 2.4.4、Pillow 12.2.0、Trimesh 4.12.2、CUDA PyTorch 2.11.0、MapAnything 1.1.2 commit `c845b8f4f6cde0c20aecd87573656c3f69f5b2b0`、checkpoint `facebook/map-anything-apache`。

## Global Constraints

- 只修改 `run.py`、`test_run.py`、`README.md`；`run.bat` 沿用預設值，不新增 runtime 檔案或 dependency。
- 不修改既有檔案編碼；UTF-8 與 UTF-8 BOM 狀態必須維持，中文終端若亂碼先執行 `chcp 65001`。
- `--quality-filter` 只允許 `balanced`、`off`，預設 `balanced`。
- `--blur-threshold` 預設 `1.5`、允許 `0～100`；`--duplicate-hamming-threshold` 預設 `4`、允許 `0～16`。
- `--confidence-percentile` 預設 `10`、允許 `0～50`；`--max-output-views` 預設 `10`、允許 `6～10`。
- `balanced` 使用 MapAnything `apply_confidence_mask=True`、指定 percentile、`use_multiview_confidence=False`。
- `off` 仍逐檔 decode/EXIF 驗證，但不篩圖、不套 confidence、不限制輸出視角，並保持舊 geometry 行為。
- 過濾後或 confidence 後少於 3 個可用 views 必須失敗；3～5 個成功但加入 `fewer_than_six_export_views` warning。
- 輸出仍只有自包含 `scene.glb` 與診斷用 `report.json`；不得加入人物模型、fusion、multi-view confidence、Cesium 或 3DGS。
- `warm_preview_seconds = quality_filter_seconds + image_load_seconds + inference_seconds + export_seconds`；18-view RTX 5060 Ti hard gate 為 `<=30` 秒。
- `export_seconds` 從 `model.infer` 完成後立即開始，涵蓋 depth-to-world、mask、selection、scene build、GLB write 與 validation。
- 使用既有 stdlib test module；unit tests 不下載模型、不要求 GPU。每個 task 必須先 RED、再 GREEN、再停在 reviewer checkpoint。
- 根目錄 `CLAUDE.md` 明訂 commit/push 由使用者執行；執行者不得自行 stage、commit 或 push，只回報每個 checkpoint 的 diff 與驗證證據。
- 所有測試命令從 `D:\mytools\Image_to_Mesh_web` 執行，避免 `mapanything_poc` package import 路徑錯誤。

## File Map

- `run.py`：保留現有單檔 CLI；新增品質分析、view summary/selection、分階段 report、原子 GLB helper，並接回現有 inference/export flow。
- `test_run.py`：沿用 `RunHelpersTest`，以 Pillow/NumPy synthetic data、fake model 與 `unittest.mock` 覆蓋所有非 GPU 決策。
- `README.md`：完成後更新 CLI、balanced/off、report v2、30 秒定義與仍不會移除人物的限制。
- `.superpowers/sdd/render_glb.py`：只在 Task 6 驗收時加入 reference-bounds 參數，確保 off/balanced 使用相同四組相機；它仍是 ignored validation helper，不屬於 runtime。
- `D:\mytools\Image_to_Mesh_web\history.md`：每個重要決策／實測結果完成後，以最小 patch 記錄；不納入 nested repo commit。

---

### Task 1: Add the pre-inference image quality gate

**Files:**
- Modify: `test_run.py:1-234`
- Modify: `run.py:13-332`

**Interfaces:**
- Produces: `sharpness_score(gray: numpy.ndarray) -> float`
- Produces: `dhash64(gray: PIL.Image.Image) -> int`
- Produces: `analyze_image_quality(path: Path, *, enabled: bool) -> tuple[float | None, int | None]`
- Produces: `filter_input_images(images: list[Path], *, mode: str, blur_threshold: float, duplicate_hamming_threshold: int) -> tuple[list[Path], list[dict]]`
- Consumes: existing natural-order `list[Path]` from `discover_images()`.

- [ ] **Step 1: Add RED tests for image metrics and EXIF validation**

Add `from unittest import mock` and `from PIL import ImageOps` to `test_run.py`, then add:

```python
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
```

- [ ] **Step 2: Add RED tests for blur boundary and fixed-anchor dHash grouping**

```python
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
    self.assertTrue(all(call.kwargs == {"enabled": False} for call in analyze.call_args_list))
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
```

- [ ] **Step 3: Run the focused tests and verify RED**

Run:

```powershell
.\mapanything_poc\.venv\Scripts\python.exe -m unittest mapanything_poc.test_run -v
```

Expected: the original nine tests pass and every new Task 1 test fails because the new helpers do not exist.

- [ ] **Step 4: Implement the metric helpers in `run.py`**

Add `import math` beside the stdlib imports, constants below `SUPPORTED_EXTENSIONS`, and these helpers before `discover_images()`:

```python
DEFAULT_BLUR_THRESHOLD = 1.5
DEFAULT_DUPLICATE_HAMMING_THRESHOLD = 4
DEFAULT_CONFIDENCE_PERCENTILE = 10.0
DEFAULT_MAX_OUTPUT_VIEWS = 10


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
```

- [ ] **Step 5: Implement fixed-anchor filtering in `run.py`**

```python
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
```

- [ ] **Step 6: Verify GREEN and stop at the Task 1 review checkpoint**

```powershell
.\mapanything_poc\.venv\Scripts\python.exe -m unittest mapanything_poc.test_run -v
.\mapanything_poc\.venv\Scripts\python.exe -m py_compile mapanything_poc\run.py mapanything_poc\test_run.py
git -C mapanything_poc diff --check
git -C mapanything_poc status --short
```

Expected: all tests pass, compilation and diff check exit `0`, and the unstaged diff contains only `run.py` and `test_run.py`; wait for user review/commit before Task 2.

---

### Task 2: Add confidence summary and representative-view selection

**Files:**
- Modify: `test_run.py`
- Modify: `run.py`

**Interfaces:**
- Consumes: MapAnything model object and preloaded `views`.
- Produces: `infer_views(model, views, *, mode: str, confidence_percentile: float) -> list[dict]`
- Produces: `summarize_view(mask, confidence, *, require_quad: bool = True) -> dict`
- Produces: `select_representative_indices(scores: list[float], max_output_views: int) -> list[int]`
- Produces: `select_export_view_indices(view_reports: list[dict], *, mode: str, max_output_views: int) -> list[int]`
- Later Task 4 maps the relative selected indices back to reconstruction/report indices.

- [ ] **Step 1: Add RED tests for exact MapAnything kwargs**

```python
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
```

- [ ] **Step 2: Add RED tests for empty/no-quad/usable masks**

```python
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
```

- [ ] **Step 3: Add RED tests for the exact temporal-bin formula**

```python
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
    self.assertTrue(all(reports[index]["exclusion_reason"] is None for index in selected))

    too_few = [dict(report, exported=False) for report in reports[:2]]
    with self.assertRaisesRegex(ValueError, "at least 3"):
        run.select_export_view_indices(
            too_few, mode="balanced", max_output_views=10
        )
    self.assertFalse(any(report["exported"] for report in too_few))
```

- [ ] **Step 4: Run the focused tests and verify RED**

```powershell
.\mapanything_poc\.venv\Scripts\python.exe -m unittest mapanything_poc.test_run -v
```

Expected: Tasks 1 tests pass and every new Task 2 test fails with `AttributeError` for the missing helpers.

- [ ] **Step 5: Implement the inference seam and view summary**

Add before `_run_inference()`:

```python
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
```

- [ ] **Step 6: Implement deterministic temporal-bin selection**

```python
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
```

- [ ] **Step 7: Verify GREEN and stop at the Task 2 review checkpoint**

```powershell
.\mapanything_poc\.venv\Scripts\python.exe -m unittest mapanything_poc.test_run -v
.\mapanything_poc\.venv\Scripts\python.exe -m py_compile mapanything_poc\run.py mapanything_poc\test_run.py
git -C mapanything_poc diff --check
git -C mapanything_poc status --short
```

---

### Task 3: Add report v2, validated CLI options, and transactional output

**Files:**
- Modify: `test_run.py`
- Modify: `run.py`

**Interfaces:**
- Produces: `validate_options(*, mode: str, blur_threshold: float | str, duplicate_hamming_threshold: int | str, confidence_percentile: float | str, max_output_views: int | str) -> tuple[float, int, float, int]`
- Produces: `prepare_output(output_dir: Path) -> None`
- Produces: `finalize_timings(timings: dict[str, float]) -> None`
- Produces: `validate_completed_report(report: dict) -> None`
- Produces: `export_scene_atomic(scene, pending_asset: dict, output_dir: Path, expected_views: int) -> dict`
- Preserves: `write_report()` atomic replacement and the existing report when a new non-finite payload is rejected.

- [ ] **Step 1: Add RED tests for CLI defaults/ranges and schema v2**

```python
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
```

- [ ] **Step 2: Add RED tests for timing formulas and atomic report retention**

```python
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
    report["asset"]["image_count"] = 2
    with self.assertRaisesRegex(ValueError, "asset counts"):
        run.validate_completed_report(report)
```

- [ ] **Step 3: Add RED tests for stale-output cleanup and atomic GLB publication**

```python
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
```

Also update the existing textured GLB expected dictionaries/assertions to include `"image_count": 1`.

- [ ] **Step 4: Run the focused tests and verify RED**

```powershell
.\mapanything_poc\.venv\Scripts\python.exe -m unittest mapanything_poc.test_run -v
```

Expected: failures identify missing parser attributes/helpers and schema version `1`.

- [ ] **Step 5: Implement CLI validation, schema v2, and report cleanup**

Replace `base_report()`, harden `write_report()`, and add the pure utilities:

```python
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
    if (
        not 0 <= duplicate_hamming_threshold <= 16
    ):
        raise ValueError("duplicate_hamming_threshold must be between 0 and 16")
    if not math.isfinite(confidence_percentile) or not 0 <= confidence_percentile <= 50:
        raise ValueError("confidence_percentile must be finite and between 0 and 50")
    if (
        not 6 <= max_output_views <= 10
    ):
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
    if any(
        row.get("rejection_reason") not in {None, "blurry", "near_duplicate"}
        for row in decisions
    ):
        raise ValueError("quality rejection reason is invalid")
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
    if quality.get("mode") == "off" and any(
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
```

Extend `_parser()` with:

```python
parser.add_argument("--quality-filter", choices=("balanced", "off"), default="balanced")
parser.add_argument("--blur-threshold", default=DEFAULT_BLUR_THRESHOLD)
parser.add_argument(
    "--duplicate-hamming-threshold",
    default=DEFAULT_DUPLICATE_HAMMING_THRESHOLD,
)
parser.add_argument(
    "--confidence-percentile", default=DEFAULT_CONFIDENCE_PERCENTILE
)
parser.add_argument(
    "--max-output-views", default=DEFAULT_MAX_OUTPUT_VIEWS
)
```

- [ ] **Step 6: Implement atomic scene publication and image counts**

Add `"image_count": view_count` to `build_textured_scene()`'s pending asset and `"image_count": expected_views` to `inspect_textured_glb()`'s return value. Then add:

```python
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
```

- [ ] **Step 7: Verify GREEN and stop at the Task 3 review checkpoint**

```powershell
.\mapanything_poc\.venv\Scripts\python.exe -m unittest mapanything_poc.test_run -v
.\mapanything_poc\.venv\Scripts\python.exe -m py_compile mapanything_poc\run.py mapanything_poc\test_run.py
git -C mapanything_poc diff --check
git -C mapanything_poc status --short
```

---

### Task 4: Wire the quality-filtered reconstruction pipeline

**Files:**
- Modify: `test_run.py`
- Modify: `run.py:333-504`

**Interfaces:**
- Consumes: all helpers from Tasks 1–3.
- Replaces: `_run_inference(images, output_dir) -> dict`.
- Produces: `_run_inference(images, output_dir, report, *, mode, confidence_percentile, max_output_views) -> None`.
- Produces: a staged schema-v2 report and one validated GLB containing only selected views.

- [ ] **Step 1: Add RED tests for quality-stage failure and stale-scene removal**

Add this test helper and test methods:

```python
def _write_valid_input_images(self, count=3):
    paths = []
    for index in range(count):
        path = self.input_dir / f"IMG_{index}.png"
        Image.new("RGB", (8, 8), (index * 30, 20, 10)).save(path)
        paths.append(path)
    return paths

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
    with mock.patch.object(
        run, "filter_input_images", return_value=(paths[:2], decisions)
    ), mock.patch.object(run, "_run_inference") as inference:
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

def test_main_malformed_numeric_option_writes_failed_report(self):
    exit_code = run.main(
        [
            "--input-dir",
            str(self.input_dir),
            "--output-dir",
            str(self.output_dir),
            "--confidence-percentile",
            "not-a-number",
        ]
    )
    self.assertEqual(exit_code, 1)
    report = json.loads((self.output_dir / "report.json").read_text("utf-8"))
    self.assertEqual(report["status"], "failed")
    self.assertEqual(report["error"]["class"], "ValueError")

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

    with mock.patch.object(
        run, "filter_input_images", return_value=(paths, decisions)
    ), mock.patch.object(run, "_run_inference", side_effect=fail_after_checkpoint):
        exit_code = run.main(
            ["--input-dir", str(self.input_dir), "--output-dir", str(self.output_dir)]
        )
    self.assertEqual(exit_code, 1)
    report = json.loads((self.output_dir / "report.json").read_text("utf-8"))
    self.assertEqual(report["error"]["message"], "postprocess failed")
    self.assertEqual(len(report["views"]), 3)
    self.assertEqual(report["quality_filter"]["usable_view_count"], 3)
```

- [ ] **Step 2: Add RED test for `off` report semantics**

```python
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

    with mock.patch.object(
        run, "filter_input_images", return_value=(paths, decisions)
    ), mock.patch.object(run, "_run_inference", side_effect=finish_without_gpu):
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
    quality = json.loads(
        (self.output_dir / "report.json").read_text("utf-8")
    )["quality_filter"]
    self.assertIsNone(quality["blur_threshold"])
    self.assertIsNone(quality["duplicate_hamming_threshold"])
    self.assertIsNone(quality["confidence_percentile"])
    self.assertIsNone(quality["max_output_views"])
    self.assertEqual(quality["inference_count"], 3)
```

- [ ] **Step 3: Run integration-focused tests and verify RED**

```powershell
.\mapanything_poc\.venv\Scripts\python.exe -m unittest mapanything_poc.test_run -v
```

Expected: failures show that `main()` does not call the filter, does not pass the shared report to `_run_inference`, and does not emit the v2 quality block.

- [ ] **Step 4: Replace `main()` with staged quality/report flow**

Numeric option text is intentionally left as strings by argparse, then converted inside the `try` block so malformed and out-of-range values both produce `ValueError` plus a failed report before any model import.

```python
def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = base_report([])
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
```

- [ ] **Step 5: Replace `_run_inference()` with selected-view export**

Keep the existing CUDA/environment fields verbatim, but change the signature and timer boundaries. The complete control flow must match this block:

```python
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
```

Do not move selection logic into `build_textured_scene()`: that function continues receiving only final selected arrays and remains responsible solely for geometry/material construction.

- [ ] **Step 6: Run all unit/static checks and stop at the Task 4 review checkpoint**

```powershell
.\mapanything_poc\.venv\Scripts\python.exe -m unittest mapanything_poc.test_run -v
.\mapanything_poc\.venv\Scripts\python.exe -m py_compile mapanything_poc\run.py mapanything_poc\test_run.py
git -C mapanything_poc diff --check
git -C mapanything_poc status --short
```

---

### Task 5: Document and smoke-test the complete 8-view workflow

**Files:**
- Modify: `README.md:36-146`
- Modify after verification: `D:\mytools\Image_to_Mesh_web\history.md`
- Verify unchanged: `run.bat`

**Interfaces:**
- Consumes: completed CLI from Task 4 and existing fixture `D:\mapanything_poc_runs\input8`.
- Produces: balanced/off smoke outputs and user-facing instructions; no new runtime code.

- [ ] **Step 1: Replace the outdated README quality section**

Replace `## 尚未實作的品質改善` with the following text and update the old 20-second paragraph to the same 30-second definition:

````markdown
## 快速品質模式

預設 `balanced` 會先排除明顯模糊與近重複照片，套用 MapAnything p10 learned confidence mask，再依拍攝順序最多輸出 10 個代表視角。低信心區會直接形成洞；工具仍不會補洞或融合表面。

```powershell
.\.venv\Scripts\python.exe .\run.py `
  --input-dir .\input `
  --output-dir .\output `
  --quality-filter balanced `
  --blur-threshold 1.5 `
  --duplicate-hamming-threshold 4 `
  --confidence-percentile 10 `
  --max-output-views 10
```

`--quality-filter off` 只作 A/B、除錯與低紋理場景退路：仍驗證每張圖片可解碼，但不做模糊／重複篩選、不套 confidence mask，也不限制輸出 views。

`report.json` schema v2 會列出每張照片的 sharpness、拒絕原因、代表視角決策，以及拆開的 quality/image-load/inference/export 時間。Warm preview 定義為 quality filter + 每次 image load + inference + postprocess/export；目標機器的 18-view hard gate 是 30 秒。

本模式不含人物偵測。移動人物仍應在拍攝時清場或在輸入前排除；confidence mask 只能挖掉模型不確定區域，不能保證自動消除所有手腳碎片。
````

- [ ] **Step 2: Run balanced and off against the existing 8-photo fixture**

```powershell
.\mapanything_poc\.venv\Scripts\python.exe mapanything_poc\run.py --input-dir D:\mapanything_poc_runs\input8 --output-dir D:\mapanything_poc_runs\office_8_quality_balanced
.\mapanything_poc\.venv\Scripts\python.exe mapanything_poc\run.py --input-dir D:\mapanything_poc_runs\input8 --output-dir D:\mapanything_poc_runs\office_8_quality_off --quality-filter off
```

Expected: both commands exit `0`; balanced exports 3–8 views, off exports all 8 views, and neither output contains `scene.glb.tmp` or `report.json.tmp`.

- [ ] **Step 3: Validate reports and off geometry parity**

```powershell
$balanced = Get-Content -Raw -Encoding UTF8 D:\mapanything_poc_runs\office_8_quality_balanced\report.json | ConvertFrom-Json
$off = Get-Content -Raw -Encoding UTF8 D:\mapanything_poc_runs\office_8_quality_off\report.json | ConvertFrom-Json
if ($balanced.status -ne 'success' -or $off.status -ne 'success') { throw '8-view smoke failed' }
if ($balanced.schema_version -ne 2 -or $off.schema_version -ne 2) { throw 'report schema is not v2' }
if ($balanced.asset.mesh_count -lt 3 -or $balanced.asset.mesh_count -gt 8) { throw 'balanced count out of range' }
if ($off.asset.mesh_count -ne 8 -or $off.quality_filter.mode -ne 'off') { throw 'off did not preserve all views' }
if ($null -ne $off.quality_filter.confidence_percentile) { throw 'off confidence threshold must be null' }
$balanced.timings | Format-List
$off.timings | Format-List
```

Then compare the old 8-view geometry and new `off` geometry:

```powershell
.\mapanything_poc\.venv\Scripts\python.exe -c "import numpy as np, trimesh; old=trimesh.load(r'D:\mapanything_poc_runs\office_8\scene.glb', force='scene'); new=trimesh.load(r'D:\mapanything_poc_runs\office_8_quality_off\scene.glb', force='scene'); assert len(old.geometry)==len(new.geometry)==8; assert sum(len(g.vertices) for g in old.geometry.values())==sum(len(g.vertices) for g in new.geometry.values()); assert sum(len(g.faces) for g in old.geometry.values())==sum(len(g.faces) for g in new.geometry.values()); np.testing.assert_allclose(old.bounds,new.bounds)"
```

- [ ] **Step 4: Run final static checks, update history, and stop at the Task 5 review checkpoint**

Run:

```powershell
.\mapanything_poc\.venv\Scripts\python.exe -m unittest mapanything_poc.test_run -v
.\mapanything_poc\.venv\Scripts\python.exe -m py_compile mapanything_poc\run.py mapanything_poc\test_run.py
git -C mapanything_poc diff --check
git -C mapanything_poc status --short
```

Use `apply_patch` to append the two exact report timing/count/byte results and the geometry-parity verdict under a new `2026-07-10 — MapAnything 快速品質過濾 8-view smoke` heading in `history.md`. Leave `README.md` unstaged for user review/commit.

Expected: `run.bat` has no diff because its existing command automatically receives all new defaults.

---

### Task 6: Run the restored 18-view performance and visual acceptance gate

**Files and paths:**
- Required input: `D:\mapanything_poc_runs\input18_quality_gate`
- Generated manifest: `D:\mapanything_poc_runs\input18_quality_gate\SHA256SUMS.txt`
- Generated outputs: `D:\mapanything_poc_runs\office_18_quality_off`, `D:\mapanything_poc_runs\office_18_quality_balanced_a`, `D:\mapanything_poc_runs\office_18_quality_balanced_b`
- Modify ignored validation renderer: `D:\mytools\Image_to_Mesh_web\.superpowers\sdd\render_glb.py`
- Modify after user signoff: `D:\mytools\Image_to_Mesh_web\history.md`

**Interfaces:**
- Consumes: the exact original 18 JPEGs named in the approved spec.
- Produces: immutable hash evidence, off/balanced A/B reports, four-view Blender renders, and the final user visual verdict.
- Execution gate: if any original JPEG is unavailable, mark this task blocked and stop; do not extract or substitute GLB-embedded 518×392 JPEGs.

- [ ] **Step 1: Verify the exact fixture before model execution**

```powershell
$expected = @(
  'IMG_20260626_172708.jpg','IMG_20260626_172710.jpg','IMG_20260626_172712.jpg',
  'IMG_20260626_172715.jpg','IMG_20260626_172717.jpg','IMG_20260626_172719.jpg',
  'IMG_20260626_172721.jpg','IMG_20260626_172723.jpg','IMG_20260626_172726.jpg',
  'IMG_20260626_172727.jpg','IMG_20260626_172729.jpg','IMG_20260626_172730.jpg',
  'IMG_20260626_172732.jpg','IMG_20260626_172733.jpg','IMG_20260626_172737.jpg',
  'IMG_20260626_172738.jpg','IMG_20260626_172740.jpg','IMG_20260626_172742.jpg'
)
if (-not (Test-Path -LiteralPath D:\mapanything_poc_runs\input18_quality_gate -PathType Container)) { throw '18-view fixture directory is missing' }
$sourceFiles = Get-ChildItem D:\mapanything_poc_runs\input18_quality_gate -File | Where-Object Name -ne 'SHA256SUMS.txt'
$actual = $sourceFiles | Where-Object { $_.Extension.ToLowerInvariant() -in @('.jpg','.jpeg','.png') } | Sort-Object Name | Select-Object -ExpandProperty Name
$missing = $expected | Where-Object { $_ -notin $actual }
$extra = $actual | Where-Object { $_ -notin $expected }
$nonImages = $sourceFiles | Where-Object { $_.Extension.ToLowerInvariant() -notin @('.jpg','.jpeg','.png') }
if ($missing.Count -or $extra.Count -or $nonImages.Count -or $actual.Count -ne 18) { throw "fixture mismatch; missing=$missing extra=$extra nonImages=$($nonImages.Name)" }
```

- [ ] **Step 2: Generate the SHA-256 fixture manifest**

This is a generated evidence artifact, so PowerShell may write it directly with explicit no-BOM UTF-8:

```powershell
$lines = Get-ChildItem D:\mapanything_poc_runs\input18_quality_gate -File | Where-Object Name -ne 'SHA256SUMS.txt' | Sort-Object Name | Get-FileHash -Algorithm SHA256 | ForEach-Object { "$($_.Hash.ToLowerInvariant()) *$([IO.Path]::GetFileName($_.Path))" }
[IO.File]::WriteAllLines('D:\mapanything_poc_runs\input18_quality_gate\SHA256SUMS.txt', $lines, [Text.UTF8Encoding]::new($false))
```

Expected: exactly 18 non-empty hash lines; the manifest itself is not an input image and is not committed.

- [ ] **Step 3: Run one off baseline and two balanced repetitions**

```powershell
.\mapanything_poc\.venv\Scripts\python.exe mapanything_poc\run.py --input-dir D:\mapanything_poc_runs\input18_quality_gate --output-dir D:\mapanything_poc_runs\office_18_quality_off --quality-filter off
.\mapanything_poc\.venv\Scripts\python.exe mapanything_poc\run.py --input-dir D:\mapanything_poc_runs\input18_quality_gate --output-dir D:\mapanything_poc_runs\office_18_quality_balanced_a
.\mapanything_poc\.venv\Scripts\python.exe mapanything_poc\run.py --input-dir D:\mapanything_poc_runs\input18_quality_gate --output-dir D:\mapanything_poc_runs\office_18_quality_balanced_b
```

Expected: three exit codes are `0`; off exports 18 meshes; each balanced run exports 6–10 meshes.

- [ ] **Step 4: Enforce the machine-readable acceptance gates**

```powershell
$off = Get-Content -Raw -Encoding UTF8 D:\mapanything_poc_runs\office_18_quality_off\report.json | ConvertFrom-Json
$a = Get-Content -Raw -Encoding UTF8 D:\mapanything_poc_runs\office_18_quality_balanced_a\report.json | ConvertFrom-Json
$b = Get-Content -Raw -Encoding UTF8 D:\mapanything_poc_runs\office_18_quality_balanced_b\report.json | ConvertFrom-Json
foreach ($report in @($off,$a,$b)) {
  if ($report.status -ne 'success' -or $report.schema_version -ne 2) { throw 'run/report failure' }
  if ($report.asset.mesh_count -ne $report.asset.material_count -or $report.asset.mesh_count -ne $report.asset.texture_count -or $report.asset.mesh_count -ne $report.asset.image_count) { throw 'asset counts differ' }
}
if ($off.asset.mesh_count -ne 18) { throw 'off did not export 18 views' }
foreach ($report in @($a,$b)) {
  if ($report.asset.mesh_count -lt 6 -or $report.asset.mesh_count -gt 10) { throw 'balanced view count outside 6-10' }
  if ($report.timings.warm_preview_seconds -gt 30.0) { throw 'warm preview exceeded 30 seconds' }
  if ($report.asset.glb_bytes -ge 137458984) { throw 'balanced GLB did not beat byte baseline' }
}
$namesA = @($a.views | Where-Object { $_.exported } | Select-Object -ExpandProperty filename)
$namesB = @($b.views | Where-Object { $_.exported } | Select-Object -ExpandProperty filename)
$filterA = $a.quality_filter.images | Select-Object filename,accepted_for_inference,rejection_reason,near_duplicate_of | ConvertTo-Json -Compress
$filterB = $b.quality_filter.images | Select-Object filename,accepted_for_inference,rejection_reason,near_duplicate_of | ConvertTo-Json -Compress
$viewsA = $a.views | Select-Object filename,usable_for_export,exported,exclusion_reason | ConvertTo-Json -Compress
$viewsB = $b.views | Select-Object filename,usable_for_export,exported,exclusion_reason | ConvertTo-Json -Compress
[pscustomobject]@{
  OffMeshes=$off.asset.mesh_count
  BalancedMeshesA=$a.asset.mesh_count
  BalancedMeshesB=$b.asset.mesh_count
  WarmSecondsA=$a.timings.warm_preview_seconds
  WarmSecondsB=$b.timings.warm_preview_seconds
  GlbBytesA=$a.asset.glb_bytes
  GlbBytesB=$b.asset.glb_bytes
  SelectedNamesEqual=([string]::Join('|',$namesA) -eq [string]::Join('|',$namesB))
  FilterDecisionsEqual=($filterA -eq $filterB)
  ViewDecisionsEqual=($viewsA -eq $viewsB)
} | Format-List
```

Record total face-count differences separately without making them a hard gate:

```powershell
.\mapanything_poc\.venv\Scripts\python.exe -c "import trimesh; paths=[r'D:\mapanything_poc_runs\office_18_quality_balanced_a\scene.glb',r'D:\mapanything_poc_runs\office_18_quality_balanced_b\scene.glb']; print([(path,sum(len(g.faces) for g in trimesh.load(path,force='scene').geometry.values())) for path in paths])"
```

Selected-name/face-count differences between CUDA/BF16 repetitions are recorded but are not an exact hard gate; all pure filter/selection tests remain exact.

- [ ] **Step 5: Render fixed viewpoints for off and balanced A**

The current renderer derives its camera from each GLB's own bounds, which would invalidate A/B framing. Use `apply_patch` to replace its source/output/bounds setup with this reference-GLB-aware block; leave the existing camera/light/render loop unchanged:

```python
argument_index = sys.argv.index("--")
source = Path(sys.argv[argument_index + 1])
output_dir = Path(sys.argv[argument_index + 2])
reference_source = (
    Path(sys.argv[argument_index + 3])
    if len(sys.argv) > argument_index + 3
    else source
)
output_dir.mkdir(parents=True, exist_ok=True)


def clear_objects():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


clear_objects()
bpy.ops.import_scene.gltf(filepath=str(reference_source))
reference_meshes = [
    obj for obj in bpy.context.scene.objects if obj.type == "MESH"
]
corners = [
    obj.matrix_world @ Vector(corner)
    for obj in reference_meshes
    for corner in obj.bound_box
]
minimum = Vector(
    (
        min(point.x for point in corners),
        min(point.y for point in corners),
        min(point.z for point in corners),
    )
)
maximum = Vector(
    (
        max(point.x for point in corners),
        max(point.y for point in corners),
        max(point.z for point in corners),
    )
)
center = (minimum + maximum) / 2
radius = max((maximum - minimum).length / 2, 1.0)

clear_objects()
bpy.ops.import_scene.gltf(filepath=str(source))
```

Pass the same off GLB as the reference argument to both renders:

```powershell
& 'C:\Program Files\Blender Foundation\Blender 5.0\blender.exe' --background --python 'D:\mytools\Image_to_Mesh_web\.superpowers\sdd\render_glb.py' -- 'D:\mapanything_poc_runs\office_18_quality_off\scene.glb' 'D:\mapanything_poc_runs\office_18_quality_off\renders' 'D:\mapanything_poc_runs\office_18_quality_off\scene.glb'
& 'C:\Program Files\Blender Foundation\Blender 5.0\blender.exe' --background --python 'D:\mytools\Image_to_Mesh_web\.superpowers\sdd\render_glb.py' -- 'D:\mapanything_poc_runs\office_18_quality_balanced_a\scene.glb' 'D:\mapanything_poc_runs\office_18_quality_balanced_a\renders' 'D:\mapanything_poc_runs\office_18_quality_off\scene.glb'
```

Expected: each render directory contains `view_1.png` through `view_4.png`, all 1024×768. Present the two four-image sets side by side; user acceptance requires at least 3/4 balanced views to show fewer peripheral fragments/ghost layers without making the central scene unrecognizable.

- [ ] **Step 6: Record final evidence and run the completion checks**

After user visual signoff, use `apply_patch` to append the manifest path, exact counts/timings/bytes, two-run difference result, four-render verdict, and remaining holes/non-fusion limitation to `history.md`. Then run:

```powershell
.\mapanything_poc\.venv\Scripts\python.exe -m unittest mapanything_poc.test_run -v
.\mapanything_poc\.venv\Scripts\python.exe -m py_compile mapanything_poc\run.py mapanything_poc\test_run.py
git -C mapanything_poc diff --check
git -C mapanything_poc status --short
```

Expected: tests and compilation pass, diff check is empty, `git status` lists only reviewed intentional unstaged files awaiting the user's commit decision, and no claim is made for fusion, automatic person removal, Cesium production placement, or measurement accuracy.
