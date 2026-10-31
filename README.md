# BRP ChromaKey2

AI green-screen keying for Windows that runs on any modern GPU: NVIDIA, AMD or Intel, including integrated graphics.

BRP ChromaKey2 wraps [CorridorKey 2.0](https://github.com/nikopueringer/CorridorKey2.0), Corridor Digital's neural keyer, in a native desktop app. Point it at green-screen footage and it produces a clean alpha matte and a despilled foreground, frame by frame, with stable edges and hair across time. Meta's SAM models can give it a hint about what the subject is, and garbage/despill masks clean up the rest.

The app is written in Rust. It doesn't need Python, CUDA or a PyTorch install, and it works without an internet connection after the first start.

## Features

- **Neural keying with CorridorKey 2.0** at the footage's own resolution (720p to 4K and beyond) with temporal consistency across frames.
- **Subject segmentation with SAM 3.1 or SAM 2.1.** Click points, draw boxes or type what you want ("person", "the red umbrella"). Masks track forward and backward from any frame you prompt. Several **layers** per scene can each keep or remove something, and the combined mask can guide the keyer or be exported as PNGs.
- **Prompts in Brazilian Portuguese.** Text prompts are translated to English on the CPU behind the scenes ("pessoa de chapéu" just works). A small line under the prompt shows the English version.
- **Automatic scene detection** (TransNetV2). Scenes can be split, merged, enabled, disabled and given their own settings.
- **Garbage and despill masks**: bezier shapes with keyframes, feathering, add/subtract/intersect modes and per-scene stacks.
- **Live despill** (0–100%) that recomposites already-calculated frames instantly.
- **Timeline** with In/Out work area, frame stepping, scene bars and a RAM cache of finished frames for real-time playback.
- **Export** a single frame or a range as ProRes 4444 with alpha, PNG sequence or linear-light EXR.
- **Portable projects**: one `.chroma` file holds the calculated frames and masks, and can embed the source media too.
- **Undo/redo** everywhere, GPU selection, and a memory-aware **Auto** mode that sizes the work to your graphics card.
- **Updates itself**: on start it checks this repository's latest release and, if it is newer, downloads it (about 35 MB) and installs it after a restart. Models download once, on first start.

## Download

Get the latest build from the [Releases](https://github.com/brurpo/BRP-ChromaKey2/releases) page. Unzip it to a folder you can write to and run `BRP-Chroma.exe`. On first start the app offers to download its models (about 9 GB) from [Hugging Face](https://huggingface.co/brurpo/CorridorKey-BRP-ONNX). Every file is checked against a pinned SHA-256 hash.

**System requirements:** Windows 10 or 11 (64-bit) and a DirectX 12 GPU. 8 GB of video memory is comfortable for 1080p; integrated GPUs work, just slowly. You'll also want 16 GB of RAM and about 10 GB of free disk space.

## How it works

CorridorKey and SAM ship as PyTorch research code that targets NVIDIA CUDA. Getting them to run natively, quickly and on every GPU vendor took several layers of work.

**One portable GPU backend.** All inference goes through ONNX Runtime with the DirectML execution provider. DirectML sits on top of Direct3D 12, so the same binaries run on NVIDIA, AMD and Intel hardware with no vendor runtime to install. The models were exported from the unchanged upstream PyTorch code to ONNX with dynamic width, height and clip length. Their original weights are kept. Compute runs in mixed FP16, while numerically sensitive parts (normalization, softmax, positional terms) stay in FP32.

**A custom executor for CorridorKey.** The original attention code is full of loops and conditionals. As ONNX control flow, they stop DirectML from fusing anything, so a frame turned into tens of thousands of tiny GPU dispatches and host round-trips. Only about 5% of the time was spent on the model's actual math. Re-exporting Stage1/Stage2 as loop-free graphs didn't fit in memory at 1080p, so the app now compiles its own plan per job:

1. Bake the clip's geometry into a symbolic-shape graph.
2. Fold its constants with ONNX Runtime's DirectML optimizer.
3. Lower the result to about 3,500 DirectML operators that share one memory heap, with buffer lifetimes planned up front.

A small C++ executor (`native/executor`) records and replays those operators with no ONNX Runtime session at all. On a 16-frame 1080p shot this halved render time (42 s → 21 s) and cut peak GPU memory from 14.4 GB to 6.9 GB. A 4K shot went from 11 minutes to about 1. The final alpha stays within 2/255 of the reference. If anything fails, the app falls back to the regular ONNX Runtime path automatically.

**Built for every GPU.** On first run, a one-operator self-test checks each GPU/driver pair and turns off DirectML "metacommands" where they are numerically wrong. One AMD integrated GPU, for example, returned NaN. Very large operators are split into chunks so integrated GPUs don't hit the Windows GPU timeout. Auto mode picks the temporal window (8, 6 or 4 frames) from the measured free video memory.

**SAM 3.1 without Python.** The image encoder, detector, text encoder and video tracker were exported to ONNX as separate graphs, and the tracking logic around them was ported to Rust:

- memory bank
- prompt handling
- forward/backward propagation
- mask post-processing

Results match upstream PyTorch on test footage (mask IoU ≥ 0.999 in FP16). The text encoder runs in FP32 on the CPU (45 ms per prompt, no video memory). A merged per-frame graph and on-GPU mask encoding bring tracking to about 0.55 s per frame within a 5.5 GB memory peak. The CLIP tokenizer is reimplemented in Rust.

**Portuguese prompts.** A resident helper detects the language with `lingua` (English and Portuguese only). It then translates with Opus-MT, converted to ONNX with INT8 weights and run on the CPU at about 16 ms per prompt. Its SentencePiece tokenizer is reimplemented in Rust too.

**Everything else is native too.** The interface uses egui on wgpu. Video goes in and out through a slim FFmpeg build. Scene detection runs TransNetV2 through the same ONNX Runtime backend. Rendering runs in a separate worker process (`ck-native.exe`), so a GPU fault can never take the editor down with it.

## Building from source

See [BUILDING.md](BUILDING.md). In short: Windows 10/11 x64, PowerShell, and about 10 GB of disk. The scripts download a pinned Rust toolchain, LLVM-MinGW and the ONNX Runtime/DirectML redistributables into the checkout itself.

## Repository layout

| Path | What's there |
| --- | --- |
| `crates/ck-desktop` | The editor (egui): timeline, viewer, masks, SAM layers, exports, updater UI |
| `crates/ck-desktop-core` | Projects, jobs, model downloads, updater logic |
| `crates/ck-native` | The worker: CorridorKey pipeline, native plan compiler, SAM 2.1/3.1 hosts, translation, scene detection |
| `crates/ck-core` | Shared image, color, mask and resampling code |
| `native/executor` | C++ DirectML executor for compiled CorridorKey plans |
| `assets/` | The pinned list of model files the app downloads |
| `licenses/` | Third-party and model license texts |
| `tools/` | Build, packaging and release scripts |

## License

The code in this repository is released under the [MIT License](LICENSE): use it, change it and share it, as long as you keep the copyright notice.

**The models are a different story.** CorridorKey 2.0 is licensed for **non-commercial use only** (CC BY-NC-SA 4.0 plus Corridor Digital's additional terms), and that applies to anything you key with this app. SAM 3.1, ONNX Runtime, DirectML and FFmpeg have their own terms too. Read [LICENSING.md](LICENSING.md) before you distribute the app or use it for paid work.

## Credits

- CorridorKey 2.0 by Corridor Digital (Niko Pueringer and team), together with the foundation models it builds on: RVM (Google DeepMind), C-RADIOv4 (NVIDIA) and MoGe-2 (Microsoft).
- SAM 2.1 and SAM 3.1 by Meta AI.
- Opus-MT by the Language Technology Research Group at the University of Helsinki.
- TransNetV2 by Tomáš Souček and Jakub Lokoč.
- ONNX Runtime and DirectML by Microsoft; FFmpeg and x264 by their contributors; egui by Emil Ernerfeldt and contributors.

This project is not affiliated with or endorsed by any of them.
