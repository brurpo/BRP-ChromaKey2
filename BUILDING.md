# Building BRP ChromaKey2

The app builds on Windows with PowerShell. Every tool is downloaded into `.tools/` inside your checkout, so nothing is installed system-wide and deleting the folder removes everything.

## Requirements

- Windows 10 or 11, 64-bit, with Windows PowerShell 5.1 or PowerShell 7
- About 10 GB of free disk space (toolchain, build output and runtime)
- An internet connection for the first setup
- For FFmpeg only: Linux or WSL with `mingw-w64`, `make`, `curl` and `pkg-config` (see step 5; you can skip it by reusing the FFmpeg files from a release)

You don't need Visual Studio, Python or CUDA.

## Steps

Run these from the repository root.

**1. Toolchain:** Rust 1.99.1 (GNU host) and LLVM-MinGW.

```powershell
.\tools\setup-toolchain.ps1
```

**2. Inference runtime:** ONNX Runtime 1.23.0 (DirectML build), DirectML 1.15.4 and the Visual C++ runtime, from Microsoft's official packages. The script checks every DLL against the hashes the app was tested with, and also unpacks the DirectML headers needed in step 4.

```powershell
.\tools\fetch-runtime.ps1
```

**3. Build the app and the worker.** Load the toolchain into this shell first. Do this again in every new shell.

```powershell
. .\tools\env.ps1
cargo build --release -p ck-desktop -p ck-native
```

**4. Build the DirectML plan executor** (`ck_plan_executor.dll`):

```powershell
.\tools\build-executor.ps1
```

**5. FFmpeg.** The app uses a small GPL build of FFmpeg (pinned commit, libavcodec 63) with x264 and zlib. To build it, run this under Linux or WSL:

```sh
sh tools/build-media.sh /tmp/media-build "$PWD/.tools/media"
```

Or download a release zip and copy its `media` folder to `.tools/media/bin`.

**6. Package:**

```powershell
.\tools\package.ps1 -MediaDir .tools\media -Version 2026.11.1.1
```

This creates `dist\BRP-ChromaKey2` and `dist\BRP-ChromaKey2-<version>-windows-x64.zip`. The folder has `BRP-Chroma.exe`, the worker, the runtime, FFmpeg, the license texts, `inventory.json` and `assets-manifest.json`. Run `BRP-Chroma.exe` from there. On first start it downloads the models listed in `assets/windows-x86_64-directml.json`.

## Tests and checks

```powershell
. .\tools\env.ps1
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Tests that need models, footage or a GPU are marked `#[ignore]` or skip themselves when their files are missing.

## Releases

`tools/publish-release.ps1` publishes the zip from step 6 as a GitHub release with that zip as its only asset. Installed copies compare their `release.json` with the latest release's tag. When it is newer, they download the zip, check it against the SHA-256 digest GitHub publishes for the asset, and replace their files, so any older version updates straight to the newest one.

```powershell
$env:GITHUB_TOKEN = '<token with contents:write>'
.\tools\publish-release.ps1 -Version 2026.11.1.1 -NotesFile notes.md
```


## Models

The ONNX model files are hosted at [brurpo/CorridorKey-BRP-ONNX](https://huggingface.co/brurpo/CorridorKey-BRP-ONNX) and pinned by commit in `assets/windows-x86_64-directml.json`, with each file's size and SHA-256. They are conversions of the original PyTorch checkpoints (see that page and `licenses/models` for sources and licenses). The export scripts are not part of this repository.
