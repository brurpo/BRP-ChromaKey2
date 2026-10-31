# Licensing

BRP ChromaKey2 combines our own code with models, libraries and tools from other people. Each part keeps its own license. This page sums them up. It isn't legal advice; the full texts in `licenses/` are what count.

## Our code: MIT

Everything in this repository that we wrote is under the [MIT License](LICENSE). That covers the Rust crates, the C++ DirectML executor and the build scripts. You can use, modify and redistribute it, commercially or not, as long as you keep the copyright and license notice.

## The models

The app downloads its models from [brurpo/CorridorKey-BRP-ONNX](https://huggingface.co/brurpo/CorridorKey-BRP-ONNX). They are ONNX conversions of other people's work and **are not covered by the MIT license**.

| Model | Used for | License |
| --- | --- | --- |
| CorridorKey 2.0 (Corridor Digital) | Keying | **Corridor Key Licence 2.0**: CC BY-NC-SA 4.0 plus additional terms. Non-commercial; derivatives must use the same license. |
| C-RADIOv4-SO400M (NVIDIA) | CorridorKey foundation features | NVIDIA Open Model License |
| MoGe-2 ViT-B (Microsoft) | CorridorKey foundation features | MIT (DINOv2 parts: Apache 2.0) |
| RVM small16 (Google DeepMind) | CorridorKey foundation features | Apache 2.0 code, CC BY 4.0 weights |
| SAM 2.1 Hiera Small (Meta) | Segmentation | Apache 2.0 |
| SAM 3.1 (Meta) | Segmentation, text prompts | SAM License: the license text must accompany redistributed copies |
| Opus-MT ROMANCE-en (University of Helsinki) | Portuguese prompt translation | Apache 2.0 |
| TransNetV2 | Scene detection | MIT |

The practical consequence: **CorridorKey 2.0 may only be used for non-commercial purposes**, so keying with this app is non-commercial unless you have separate permission from Corridor Digital. The CorridorKey graphs we publish (including the re-exported "native" Stage1/Stage2 graphs) are derivatives and keep the Corridor Key Licence 2.0.

Parts of the keying pipeline in `crates/ck-native` (input preparation, temporal windows and output decoding) were written to reproduce CorridorKey 2.0's reference implementation. If you plan to redistribute or commercialize a modified version, review the Corridor Key Licence's terms on derivative works first.

Full texts, publishers, pinned upstream revisions and original checkpoint hashes are in `licenses/models/` (`NOTICE.txt` and `MODEL_LICENSES.json`).

## Runtime libraries shipped in release builds

| Component | License | Notes |
| --- | --- | --- |
| ONNX Runtime 1.23.0 | MIT | `licenses/runtime/ONNXRUNTIME_*` |
| DirectML 1.15.4 | Microsoft DirectML license (redistributable) | `licenses/runtime/DIRECTML_*` |
| Visual C++ runtime 14.39 | Microsoft Visual C++ Runtime license | `licenses/runtime/MICROSOFT_VISUAL_CPP_RUNTIME_LICENSE.txt` |
| FFmpeg, x264, zlib | **GPL v2** (FFmpeg is built with `--enable-gpl` for x264), zlib license | Source archives for the exact build are attached to every release |
| MinGW-w64 winpthreads | MIT-style | `licenses/runtime/MINGW_RUNTIME_COPYRIGHT.txt` |

Because the bundled FFmpeg is a GPL build, a release package as a whole is distributed under GPL v2-compatible terms. Our MIT code is compatible with that. For the GPL parts, the corresponding source is the `ffmpeg-source` archives attached to each release and `tools/build-media.sh`, which rebuilds them.

## Rust dependencies

The crates the app is built from are listed with their licenses and full license texts in `licenses/THIRD_PARTY_RUST_NOTICES.txt`, which `tools/collect-rust-notices.ps1` generates from `Cargo.lock`. Almost all of them are MIT and/or Apache 2.0. The bundled egui fonts are under the licenses in `licenses/fonts/`. A few crates whose published archives lack a license file have it pinned from their source repository in `licenses/pinned/`.

The small CUDA kernels in `crates/ck-native/kernels` come from PyTorch (BSD-style, `LICENSE-PYTORCH.txt`).
