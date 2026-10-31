# FFmpeg sources for BRP ChromaKey2 releases

BRP ChromaKey2 release zips include a small GPL build of FFmpeg (with x264 and zlib) in their `media` folder. This branch holds the exact sources and configuration of that build, as the GPL requires.

| File | What it is |
| --- | --- |
| `ffmpeg.tar.gz` | FFmpeg source, commit `946fcce07b` |
| `x264.tar.gz` | x264 source (stable branch at build time) |
| `zlib.tar.gz` | zlib 1.3.1 source |
| `config.h`, `config.mak`, `config_components.h` | FFmpeg's generated build configuration |
| `archive-sha256.txt` | SHA-256 of the three source archives |
| `native-sha256.txt` | SHA-256 of the built DLLs and executables |

`tools/build-media.sh` on the `main` branch rebuilds the binaries from these sources. FFmpeg and x264 are licensed under the GNU GPL v2 or later, and zlib under the zlib license. The license texts are in `licenses/runtime` on `main` and in every release zip.