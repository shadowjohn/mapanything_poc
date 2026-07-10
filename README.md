# MapAnything 多照片快速 3D 預覽

這是一個獨立、僅使用 CUDA 的命令列工具。輸入同一場景的多張連續照片後，使用 MapAnything 估算相機
姿態與每個視角的場景幾何，輸出一個含照片貼圖的 `scene.glb`。

本工具適合快速確認場景的大致形狀、涵蓋角度與相機對位，不是融合式攝影測量，也不提供測量級精度。

## 適合與不適合的情境

適合：

- 工地、涵洞、房間、機房、倉庫等靜態整體場景。
- 同一裝置、同一次、短時間連續拍攝的照片。
- 希望在數十秒內取得可旋轉查看的粗略 3D 預覽。

不適合：

- 人物、動物或移動車輛的 3D 掃描。
- 跨日期、不同施工階段或不同裝置混合的照片。
- 需要封閉、無縫、可量測或可直接施工驗收的 mesh。

## 系統需求

- Windows 與 Python 3.12。
- NVIDIA GPU 與 CUDA；沒有 CPU fallback。
- 模型第一次下載需要網路與足夠的磁碟快取空間。

## 安裝

```powershell
git clone https://github.com/shadowjohn/mapanything_poc.git
Set-Location .\mapanything_poc

py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
New-Item -ItemType Directory -Force .\input | Out-Null
```

## 執行

最簡單的方式：

1. 將至少 3 張 `.jpg`、`.jpeg` 或 `.png` 照片放入 `input/`。
2. 執行 `run.bat`。
3. 到 `output/` 取得 `scene.glb` 與 `report.json`。

也可以直接使用命令列：

```powershell
.\.venv\Scripts\python.exe .\run.py --input-dir .\input --output-dir .\output
```

## 拍照方法

### 基本原則

- 每個專案必須是同一次、同一裝置、短時間連續拍攝，且場景結構沒有改變。
- 建議先用 8～30 張測試；目前在目標機器實測到 18 張，不代表更大圖集一定能維持相同速度。
- 相鄰照片維持約 70%～85% 的內容重疊，讓同一牆角、柱子或設備連續出現在多張照片中。
- 人要帶著相機移動，產生左右與前後視差；不要只站在原地旋轉手機拍全景。
- 每次前進一小段或轉約 10～15 度再拍一張，避免視角跳太遠。
- 大型場景可先拍一圈平視角度，再以稍高或稍低的位置補第二圈。
- 固定同一顆鏡頭與焦段，不要途中切換超廣角、主鏡頭、數位變焦或人像模式。
- 保持照片清楚、曝光穩定；移動時先停穩再按快門，刪除明顯模糊的照片。

### 建議拍攝順序

1. 先以同一顆鏡頭拍一張能辨識整體範圍的建立照。
2. 沿著場景外圍緩慢移動，每張都保留前一張的大部分特徵。
3. 對牆角、門框、柱子、標誌或設備邊緣多保留幾個連續角度，這些特徵有助於相機對位。
4. 完成主要環繞後，再補被遮住的角落與不同高度；不要突然跳到沒有重疊的新區域。
5. 執行前先刪除模糊、重複、手指遮鏡頭或明顯拍到移動物體的照片。

### 如何避免人物變成「一堆腿」

目前輸出不是單一融合表面，而是把每張照片各自估算的 mesh 對齊後一起顯示。人物只要在連拍過程中改變
姿勢或位置，每個視角就可能重建出不同的手、腳與身體，最後全部疊在同一個 GLB 裡。

要降低這類問題：

- 拍攝整體場景時，盡量等人員、車輛與機具離開畫面再拍。
- 不要讓攝影者的手指、手臂、鞋子或手機殼進入鏡頭。
- 無法清場時，先完成無人的主要環繞，再把有人員的照片排除，不要混入同一批重建。
- 若人物本身是主要拍攝目標，請改用專門的人體掃描或 Gaussian Splatting 流程；本工具不適合人物建模。
- 即使人物完全不動，手指、頭髮與四肢等細長結構仍比牆面或大型設備容易破碎。

### 容易失敗的表面

- 玻璃、鏡子、亮面金屬與水面會因反射隨視角改變。
- 全黑、全白或沒有紋理的大面積牆面不容易建立可靠對應。
- 細鐵絲、欄杆、樹葉與快速移動物體容易斷裂或產生碎片。
- 必須拍攝這些區域時，增加周圍具有紋理的固定參考物，並從更多連續角度補拍。

## 輸出內容與限制

`scene.glb` 會為每個匯出視角內嵌一張模型處理解析度的 JPEG 與一個 PBR material，不需要外部貼圖檔，
也不輸出 `COLOR_0` vertex color。

每個視角的 mesh 仍然分開存在，因此重疊、接縫、破洞與局部碎片屬於目前機制的預期限制。工具沒有做
TSDF、Poisson、mesh fusion、接縫消除或重新拓撲。

`report.json` 會記錄輸入、相機姿態、信心、有效比例、執行時間、GPU 記憶體與 GLB 結構資訊。

## 實測結果

目標機器為 RTX 5060 Ti 16 GB，已測試 3、8 與 18 個視角。最終 18-view textured run：

- Warm inference + export：16.844 秒。
- Peak allocated VRAM：6511 MiB。
- Peak reserved VRAM：8892 MiB。
- 單一自包含 GLB：137,458,984 bytes。

30 秒 hard gate 計算 quality filter、每次 image load、warm inference 與 postprocess/export，不包含模型首次
載入、上傳、排隊、下載與 viewer 解析時間。

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

## 版本與授權

PoC 固定使用 Apache-2.0 licensed MapAnything code commit
`c845b8f4f6cde0c20aecd87573656c3f69f5b2b0`，checkpoint 固定為
`facebook/map-anything-apache`。

目標實測時 Hugging Face 解析到的 weights revision 為
`00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a`，但 CLI 尚未固定該 revision；上游 DINOv2 Torch Hub `main`
也未固定，因此不同時間執行不保證 bit-level reproducible。
