# MapAnything 快速品質過濾設計

## 目的

在不加入新模型、不做逐位置融合的前提下，改善 `mapanything_poc` 疊層 mesh 的重影、拉絲、重複人物與
輸出體積。新版預設先排除明顯模糊與近重複照片，再使用 MapAnything 既有 learned confidence 挖除低信心
幾何，最後最多輸出 10 個具時序涵蓋且信心較高的代表視角。

品質改善可以多花數秒，但目標仍是數十秒完成，不得退化為分鐘級工作。

## 基準

目前 RTX 5060 Ti 16 GB 的固定 18-view textured baseline：

- Warm inference：13.494 秒。
- Textured GLB export：3.349 秒。
- Warm inference + export：16.844 秒。
- 18 meshes、18 embedded JPEG textures。
- GLB：137,458,984 bytes。

目前只套用 non-ambiguous mask、depth validity 與 edge mask；learned confidence 只寫入 report，沒有參與
mesh mask。所有有效視角都各自輸出一層 mesh，沒有 fusion。

舊版 16.844 秒只包含 inference + export；`load_images` 被包在含模型 cold load 的 `load_seconds`，因此不等同
新版 per-request preview。新版效能門檻會把每次都發生的 image load 納入。

## 範圍

包含：

- 推論前的模糊照片與近重複照片篩選。
- MapAnything 內建 learned confidence percentile mask。
- 依自然拍攝順序與視角品質選出代表視角；上限可設為 6～10，預設 10。
- `report.json` schema v2，完整記錄每張照片的篩選、推論與匯出決策。
- 保留單一、自包含 textured `scene.glb`。
- 提供 `balanced` 預設模式與可重現舊行為的 `off` 模式。

不包含：

- 人物／車輛偵測或語意分割模型。
- Multi-view confidence、跨視角 depth reprojection 或逐位置最佳視角。
- TSDF、Poisson、mesh fusion、重新拓撲或接縫融合。
- Web UI、後端 API、Cesium 或 Gaussian Splatting 整合。
- 200～2500 張大量影像的分區與 keyframe 規模化策略。

## CLI 契約

新增參數：

```text
--quality-filter {balanced,off}   預設 balanced
--blur-threshold FLOAT            預設 1.5，允許 0～100
--duplicate-hamming-threshold INT 預設 4，允許 0～16
--confidence-percentile FLOAT    預設 10，允許 0～50
--max-output-views INT            預設 10，允許 6～10
```

`run.bat` 不增加互動，沿用預設 `balanced`、p10 與最多 10 views。

`off` 仍逐檔執行解碼與 EXIF orientation 驗證，但不計算 sharpness／dHash，也不篩選照片；它必須停用
confidence mask 與代表視角上限，維持目前所有輸入都推論、所有有效視角都匯出的 geometry 行為。此模式
只作 A/B、除錯與低紋理場景的安全退路。

無效數值在模型載入前以 `ValueError` 結束，並沿用 failed `report.json`。

## 資料流

```text
discover images
  → decode / EXIF validation
  → image quality analysis and blur / near-duplicate rejection (balanced only)
  → MapAnything inference with learned confidence mask
  → discard views without any valid quad (balanced; off fails)
  → temporal-bin representative selection
  → textured scene export
  → raw GLB validation
```

## 1. 影像品質分析

只使用已安裝的 Pillow 與 NumPy，不增加依賴。

每張照片：

1. 用 Pillow 開啟並套用 EXIF orientation；校正後任一邊小於 3 px 時，以 `image_too_small` 失敗。
2. 轉成灰階，以 Lanczos 等比例縮到最長邊 256 px，不放大原圖。
3. 轉成 `[0, 1]` float32。
4. 使用內部區域計算四鄰域 Laplacian：

```text
center = gray[1:-1, 1:-1]
up = gray[:-2, 1:-1]
down = gray[2:, 1:-1]
left = gray[1:-1, :-2]
right = gray[1:-1, 2:]
lap = 4 * center - up - down - left - right
gx = 0.5 * (right - left)
gy = 0.5 * (down - up)
gradient_energy = 0.5 * (mean(gx²) + mean(gy²))
sharpness = variance(lap) / (gradient_energy + 1e-12)
```

`up/down/left/right` 都取自與 `center` 相鄰且同尺寸的內部切片；sharpness 必須是 finite，否則以
`invalid_sharpness` 失敗。

預設以 `sharpness < 1.5` 判定為模糊；等於設定門檻時保留。完成版以固定 8 張既有辦公室照片實跑的
分數為 7.566～8.183，確認不會被預設值誤刪；先前設計階段的 1.892～2.046 估值不再作為驗收依據。
使用者可透過 CLI 校正門檻，所有分數與實際門檻都必須寫入 report；自動 threshold 校正明確不納入本版。

近重複判定使用 64-bit dHash：

1. 將同一灰階縮圖縮成 9×8。
2. 以 `right > left` 比較每列相鄰像素並以 row-major 打包 64 bits。
3. 依自然排序逐張分組；每組保留第一張照片的 dHash 作為固定 anchor，後續不因 winner 改變 anchor。
4. 新照片和所有 group anchor 比較；最小 Hamming distance 小於等於設定門檻時加入該組（預設 4），距離
   同分時選 anchor 較早的組，沒有符合者則建立新組。此規則刻意不做傳遞式合併，避免一段緩慢移動的
   照片被 chain 成一組。
5. 每組只保留 sharpness 較高者；同分保留自然排序較早者。所有 loser 的 `near_duplicate_of` 指向該組
   最終 winner，最後再把 winners 按自然順序排列。

處理順序先拒絕模糊，再處理近重複。輸入與結果均採現有 natural filename order，確保同一批資料重跑會
得到相同結果。

品質篩選後少於 3 張時直接失敗，不得為了湊數偷偷補回已拒絕照片。損壞或 Pillow 無法解碼的圖片也直接
失敗並指出 filename。

## 2. Learned confidence mask

`balanced` 模式直接使用 MapAnything 既有參數：

```python
model.infer(
    ...,
    apply_confidence_mask=True,
    confidence_percentile=confidence_percentile,
    use_multiview_confidence=False,
)
```

MapAnything 會對每個 view 各自計算 learned confidence percentile，使用嚴格的 `conf > threshold`，再與
non-ambiguous mask 及 edge mask 相交。現有 `prediction["mask"] & valid_depth` 與 `image_mesh` 會自然讓低
信心區不產生 faces、UV 或 texture geometry，不另寫第二套 confidence 演算法。

不啟用 multi-view confidence，因為它需要跨 view depth reprojection，記憶體與計算量會隨 view count 增加；
它也會產生較多離散同值 confidence，和官方嚴格 `>` percentile mask 組合時有整張清空的風險。

每個 view 先以以下規則檢查能否形成 face：

```text
quad_mask = (
    mask[:-1, :-1] & mask[1:, :-1] &
    mask[:-1, 1:] & mask[1:, 1:]
)
```

`mask` 全空時記為 `no_valid_pixels_after_confidence`；有 pixel 但 `quad_mask` 全空時記為
`no_valid_quad_after_confidence`。兩者都保留在 report，但不參與代表視角選擇。`valid_fraction` 仍記錄實際
mask 比例；全空時 `mean_valid_confidence` 為 JSON `null`，有 pixel 時只對有效 pixel 計算 finite mean；兩種
情況的 `selection_score` 都是 `null`。最後可用 view 少於 3 時整次失敗。

上述單 view 排除只適用 `balanced`。`off` 保留舊版 fail-fast 行為：任一 view 無法產生 mesh 時整次失敗，
不靜默改變 A/B geometry 集合。

## 3. 代表視角選擇

本階段不計算 pose distance。因輸入契約已要求同一裝置、短時間連續拍攝，自然排序可視為拍攝路徑順序。

每個可用 view 的品質分數：

```text
selection_score = mean_valid_confidence * valid_fraction
```

其中 `mean_valid_confidence` 是 combined mask 有效 pixels 的 confidence 平均，`valid_fraction` 是有效 pixel
數除以該 view 的 `H × W`；兩者都在 confidence、non-ambiguous、edge、valid-depth mask 後計算。

選擇規則：

- 可用 views `<= max_output_views`：全部輸出。
- 可用 views `> max_output_views`：令 `N` 為可用數、`K=max_output_views`，第 `i` 段使用半開區間
  `[floor(i*N/K), floor((i+1)*N/K))`，依自然順序切成 `K` 個非空連續區段。
- 每段選 `selection_score` 最高者；同分選自然排序較早者。
- 最終輸出順序仍按原自然順序，不按 score 排序。
- 最終只有 3～5 views 時照常輸出，但 report 加入 `fewer_than_six_export_views` warning。

這個策略保留整段拍攝路徑的涵蓋，不會讓十張高信心但角度相近的照片吃掉其他區域。它只選 whole-view
meshes，沒有宣稱處理同一表面的跨視角重疊。

## Report schema v2

保留既有 environment、timings、asset 與 camera data，`schema_version` 升為 2，新增：

```json
{
  "quality_filter": {
    "mode": "balanced",
    "blur_threshold": 1.5,
    "duplicate_hamming_threshold": 4,
    "confidence_percentile": 10.0,
    "max_output_views": 10,
    "input_count": 18,
    "inference_count": 16,
    "usable_view_count": 15,
    "export_view_count": 10,
    "warnings": [],
    "images": [
      {
        "filename": "IMG_0001.jpg",
        "sharpness": 1.92,
        "accepted_for_inference": true,
        "rejection_reason": null,
        "near_duplicate_of": null
      }
    ]
  },
  "timings": {
    "quality_filter_seconds": 1.2,
    "cold_load_seconds": 30.0,
    "image_load_seconds": 1.5,
    "load_seconds": 31.5,
    "inference_seconds": 14.0,
    "export_seconds": 2.0,
    "total_seconds": 48.7,
    "warm_preview_seconds": 18.7
  },
  "views": [
    {
      "filename": "IMG_0001.jpg",
      "mean_valid_confidence": 3.4,
      "valid_fraction": 0.68,
      "selection_score": 2.312,
      "usable_for_export": true,
      "exported": true,
      "exclusion_reason": null
    }
  ],
  "asset": {
    "mesh_count": 10,
    "material_count": 10,
    "texture_count": 10,
    "image_count": 10
  }
}
```

`inputs` 仍列出 discover 到的原始檔名；`quality_filter.images` 必須涵蓋全部 inputs；`views` 則列出實際送入
inference 的照片。成功時必須符合：

```text
input_count == len(inputs)
inference_count == count(accepted_for_inference)
export_view_count <= usable_view_count <= inference_count
mesh/material/texture/image count == export_view_count
```

`quality_filter.images[].rejection_reason` 只允許 `null`、`blurry`、`near_duplicate`；`views[].exclusion_reason`
只允許 `null`、`no_valid_pixels_after_confidence`、`no_valid_quad_after_confidence`、
`not_selected_as_representative`。只有 exported view 的 exclusion reason 是 `null`。

`off` 模式的 blur、duplicate、confidence 與 max-output threshold 欄位均為 `null`；每張圖的 `sharpness`、
`near_duplicate_of`、`rejection_reason` 也為 `null`，`accepted_for_inference=true`。其
`quality_filter_seconds` 仍包含逐檔解碼與 EXIF validation。

`cold_load_seconds` 包含依賴匯入、checkpoint 載入與模型搬到 GPU，只在 cold CLI／worker 啟動時發生；
`image_load_seconds` 單獨計算 MapAnything 每次請求都會執行的 `load_images`。保留 `load_seconds` 並定義為兩者
總和。`warm_preview_seconds` 不得把 per-request image load 藏入 cold load，定義為：

```text
quality_filter_seconds + image_load_seconds + inference_seconds + export_seconds
```

`export_seconds` 從 `model.infer` 完成後立刻開始，包含 depth-to-world、combined mask、view summary、代表視角
選擇、scene build、GLB 寫入與 validation；不得把這些 per-request 後處理留在所有 timing 欄位之外。

所有數值必須 finite，JSON 不允許 NaN 或 infinity。

Report 必須在每個已完成階段後原子更新：quality filter 完成後先寫 counts/decisions，inference 後再寫 views，
export/validation 後才寫 asset 與 success。後續階段失敗時，failed report 保留最後一個已完成階段的資料，
不得退回只剩 base report。

## 輸出契約

- 成功仍只需要 `scene.glb`；`report.json` 是診斷 sidecar，不是貼圖相依。
- 每個 selected view 對應一個 mesh、material、texture 與 embedded JPEG image。
- Primitive 只有 `POSITION`、`TEXCOORD_0` 與 material reference，不得有 `COLOR_0`。
- 不得有外部 texture URI。
- Mesh count 為 3～`max_output_views`；一般 6 張以上的有效輸入預期輸出 6～10 views。
- GLB validator 的 expected count 必須使用 selected view count，不再假設等於原始輸入數。
- `off` 模式在同一輸入上必須維持舊版 geometry counts、bounds 與全部視角輸出。
- 每次 run 開始時先移除同一 output directory 的舊 `scene.glb` 與暫存檔，避免失敗後誤拿前次結果；要保留
  舊成果時必須使用不同 output directory。
- 新結果先以明確 `file_type="glb"` 寫入 `scene.glb.tmp`，通過 validator 後才原子 replace 成
  `scene.glb`；失敗時刪除暫存檔，且不得留下本次 run 的 `scene.glb`。

## 錯誤與 fallback

- 原始支援格式少於 3 張：沿用現有失敗。
- 圖片無法解碼：失敗並記錄 filename。
- 模糊／重複過濾後少於 3 張：模型載入前失敗。
- Confidence 後可用 views 少於 3：失敗，不輸出成功狀態。
- 只有 3～5 個可用 views：成功輸出並寫 warning。
- `balanced` 造成合法低紋理場景被拒絕時，使用者可明確改用 `--quality-filter off` 重跑；程式不得自動靜默
  切回 off。
- GLB packaging 或 validator 失敗：沿用 failed report 與非零 exit code。

## 效能邊界

目標機器仍為 RTX 5060 Ti 16 GB。固定 18-view 測試的 hard gate：

```text
quality_filter + image load + warm inference + textured export <= 30 秒
```

首次 model load、上傳、下載與 viewer parsing 不計，但 quality filter 必須計入，不得藏在 load time。

已量測 confidence percentile + mask 對 18×392×518 synthetic arrays 為 0.083 秒；固定 8 張 4K 辦公室照片
的 1024px blur/dHash decode 為 1.095 秒。正式實作改用 256px thumbnail，預期品質分析與 confidence mask 合計
增加約 0.4～2.3 秒。此為預估，最終只採真實 18-view report 判定。

輸出從 18 views 限制為最多 10 views 後，預期 GLB bytes 與 export time 下降；這不是 hard ratio，必須記錄
實測數字，不先承諾固定 40%。

18-view hard gate fixture 固定放在 `D:\mapanything_poc_runs\input18_quality_gate\`，且必須正好包含：

```text
IMG_20260626_172708.jpg  IMG_20260626_172710.jpg  IMG_20260626_172712.jpg
IMG_20260626_172715.jpg  IMG_20260626_172717.jpg  IMG_20260626_172719.jpg
IMG_20260626_172721.jpg  IMG_20260626_172723.jpg  IMG_20260626_172726.jpg
IMG_20260626_172727.jpg  IMG_20260626_172729.jpg  IMG_20260626_172730.jpg
IMG_20260626_172732.jpg  IMG_20260626_172733.jpg  IMG_20260626_172737.jpg
IMG_20260626_172738.jpg  IMG_20260626_172740.jpg  IMG_20260626_172742.jpg
```

原本的 `C:\Users\johnho\Desktop\image` 目前已不存在，因此正式 A/B 前必須恢復這 18 個原始檔，第一次恢復
後建立 SHA-256 manifest 固定 fixture；既有 GLB 內的 518×392 embedded JPEG 不得冒充原始輸入。

## 測試與驗收

以現有 stdlib `unittest` 擴充，不新增 test framework。

Synthetic tests 必須覆蓋：

- EXIF orientation 後的 deterministic sharpness。
- 清楚圖保留、模糊圖拒絕、`sharpness == 1.5` 保留。
- Blur threshold `0～100`、dHash threshold `0～16` 的 CLI 邊界與 report 值一致。
- dHash distance `<=4` 歸為近重複，保留較清楚者；同分保留較早者。
- 一張圖同時接近多個 group anchor 時選距離最小者；距離同分選較早 anchor，且 fixed-anchor 分組不做
  傳遞式 chain merge。
- 過濾後少於 3 張明確失敗。
- Confidence p10 參數傳給 MapAnything，multi-view confidence 保持關閉。
- Confidence 後的 empty mask 與 no-valid-quad 分別使用指定 reason/null 數值；少於 3 views 明確失敗。
- `N <= max_output_views` 全選。
- `N > max_output_views` 依指定 floor 公式每段恰選一張，tie deterministic，最終順序自然排序。
- GLB mesh/material/texture/image counts 等於 selected count，仍無 `COLOR_0` 與外部 URI。
- `quality-filter=off` 保持舊 geometry counts 與 bounds。
- Report schema v2 counts、reason enum、warnings、分階段失敗資料、timing 公式與 finite JSON。
- 同一 output directory 的 failed rerun 不得留下舊 `scene.glb` 或本次暫存檔。
- 對固定 synthetic model outputs，quality filter、selected filenames、score 與 face counts 重跑完全一致。

真實 18-view A/B 驗收：

1. `balanced` status success，輸出 6～10 views。
2. `off` 可重現舊版 18-view geometry 與 packaging。
3. `balanced` 的 quality filter + image load + warm inference + export `<=30 秒`。
4. `balanced` GLB bytes 小於 137,458,984-byte baseline。
5. Blender 沿用 baseline 的固定四個 camera views 並排 render；人工確認至少 3/4 視角的外圍碎片／重影較
   `off` 少，且主場景沒有因挖洞而失去可辨識性，最後由使用者作質性簽核。
6. 重跑兩次並記錄 selected filenames、face counts 與 report decisions 的差異。CUDA/BF16 輸出不要求
   byte-identical 或 face-count exact；deterministic hard assertion 只套用固定 synthetic model outputs。

## 已知限制

- Confidence mask 只會刪除不可靠區域，不會補洞或建立單一融合表面。
- 時序區段假設檔名 natural order 對應連續拍攝順序；任意混入或重新命名的照片可能降低代表性。
- dHash 只處理近重複影像，不代表語意或 3D 視角相同。
- 本階段不會自動移除人物；避免「一堆腿」仍主要依賴清場與排除含移動人物的照片。
- 低紋理、反射、透明與細長結構仍可能破碎。
