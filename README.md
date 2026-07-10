# MapAnything multi-image PoC

This is an isolated CUDA-only CLI. It is not connected to, imported by, or
served from the current web app.

## Setup and run (PowerShell)

```powershell
Set-Location D:\mytools\Image_to_Mesh_web
py -3.12 -m venv .\mapanything_poc\.venv
.\mapanything_poc\.venv\Scripts\python.exe -m pip install --upgrade pip
.\mapanything_poc\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.\mapanything_poc\.venv\Scripts\python.exe -m pip install -r .\mapanything_poc\requirements.txt
.\mapanything_poc\.venv\Scripts\python.exe -m unittest mapanything_poc/test_run.py -v
.\mapanything_poc\.venv\Scripts\python.exe .\mapanything_poc\run.py --input-dir .\mapanything_poc\input --output-dir .\mapanything_poc\output
```

Place at least three direct-child `.jpg`, `.jpeg`, or `.png` files in the input
directory. A successful run writes `scene.glb` and `report.json` to the output
directory. CUDA is required; there is no CPU fallback. The first run downloads
the checkpoint from Hugging Face and therefore needs network access and
enough local cache space.

For the default folders, place the photos in `mapanything_poc/input` and run
`mapanything_poc/run.bat`; it writes to `mapanything_poc/output`.

`scene.glb` embeds one model-resolution JPEG and one PBR material per input
view. It has no external texture files and no `COLOR_0` vertex colors. The
meshes remain separate layered views, so overlaps and seams are expected; this
preview does not fuse or simplify them. The 20-second target covers warm
inference plus textured export only, excluding first model load, upload,
queueing, download, and viewer parsing.

The PoC pins the Apache-2.0-licensed MapAnything code. For model weights, only
the Apache checkpoint ID `facebook/map-anything-apache` is fixed. The target
run resolved Hugging Face revision
`00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a`, but the CLI does not pin that
revision, and upstream DINOv2 Torch Hub `main` remains unpinned, so runs are not
bit-level reproducible. The GLB contains aligned per-view meshes; it is not a
fused or guaranteed watertight mesh. Only view counts actually tested on the
target machine should be treated as supported, with no guarantee beyond those
tested counts.

Verified on the target RTX 5060 Ti 16 GB with 3, 8, and 18 views. The final
18-view textured run used 6511 MiB peak allocated VRAM and 8892 MiB peak
reserved VRAM; warm inference plus export took 16.844 seconds and produced a
137,458,984-byte self-contained GLB. This is the largest count tested, not a
promise that substantially larger projects will fit or retain the same speed.
