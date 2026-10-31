#!/bin/sh
# Cross-builds the slim FFmpeg used by the app (GPL build with x264 and zlib) on Linux/WSL
# with the mingw-w64 toolchain. Output: <output>/bin (DLLs + ffmpeg/ffprobe), licenses, source.
# Usage: sh tools/build-media.sh <build-folder> <output-folder>
set -eu
build_root=${1:-/tmp/brp-chromakey2-media}
output_root=${2:?Pass an absolute output folder}
mkdir -p "$build_root" "$output_root/bin" "$output_root/licenses" "$output_root/source"
cd "$build_root"
curl --fail --location --retry 3 https://codeload.github.com/FFmpeg/FFmpeg/tar.gz/946fcce07b -o ffmpeg.tar.gz
curl --fail --location --retry 3 https://zlib.net/fossils/zlib-1.3.1.tar.gz -o zlib.tar.gz
curl --fail --location --retry 3 https://code.videolan.org/videolan/x264/-/archive/stable/x264-stable.tar.gz -o x264.tar.gz
sha256sum ffmpeg.tar.gz zlib.tar.gz x264.tar.gz > "$output_root/source/archive-sha256.txt"
mkdir -p ffmpeg zlib x264 prefix
tar -xzf ffmpeg.tar.gz -C ffmpeg --strip-components=1
tar -xzf zlib.tar.gz -C zlib --strip-components=1
tar -xzf x264.tar.gz -C x264 --strip-components=1
cd zlib
make -f win32/Makefile.gcc PREFIX=x86_64-w64-mingw32- -j8 libz.a
mkdir -p "$build_root/prefix/include" "$build_root/prefix/lib"
cp zlib.h zconf.h "$build_root/prefix/include/"
cp libz.a "$build_root/prefix/lib/"
cd "$build_root/x264"
./configure --host=x86_64-w64-mingw32 --cross-prefix=x86_64-w64-mingw32- --prefix="$build_root/prefix" --enable-static --disable-cli --disable-opencl --enable-strip
make -j8
make install-lib-static
cd "$build_root/ffmpeg"
PKG_CONFIG_LIBDIR="$build_root/prefix/lib/pkgconfig" ./configure \
  --prefix="$output_root" --target-os=mingw32 --arch=x86_64 --cpu=generic \
  --cross-prefix=x86_64-w64-mingw32- --pkg-config=pkg-config --pkg-config-flags=--static \
  --extra-cflags="-I$build_root/prefix/include -Os" \
  --extra-ldflags="-L$build_root/prefix/lib -static-libgcc" \
  --enable-shared --disable-static --disable-doc --disable-debug --enable-small \
  --disable-network --disable-autodetect --disable-everything --enable-gpl \
  --enable-avcodec --enable-avformat --enable-avfilter --enable-swscale --enable-swresample \
  --enable-ffmpeg --enable-ffprobe --disable-ffplay --enable-w32threads --enable-zlib --enable-libx264 \
  --enable-protocol=file,pipe \
  --enable-decoders --enable-parsers --enable-demuxers \
  --enable-encoder=libx264,png,mpeg4,rawvideo,ffv1,prores_ks,pcm_s16le \
  --enable-muxer=mp4,mov,matroska,image2,rawvideo,null,framemd5 \
  --enable-filter=scale,pad,crop,format,select,trim,setpts,fps,setsar,null,buffer,buffersink,concat,transpose,hflip,vflip,rotate
make -j8
make install
cp COPYING.GPLv2 "$output_root/licenses/FFMPEG_GPLv2.txt"
cp "$build_root/x264/COPYING" "$output_root/licenses/X264_GPLv2.txt"
cp "$build_root/zlib/README" "$output_root/licenses/ZLIB_README.txt"
cp config.h config_components.h ffbuild/config.mak "$output_root/source/"
cp ffmpeg.tar.gz "$output_root/source/" 2>/dev/null || cp "$build_root/ffmpeg.tar.gz" "$output_root/source/"
cp "$build_root/zlib.tar.gz" "$build_root/x264.tar.gz" "$output_root/source/"
cp /usr/x86_64-w64-mingw32/lib/libwinpthread-1.dll "$output_root/bin/"
cp /usr/share/doc/mingw-w64-common/copyright "$output_root/licenses/MINGW_RUNTIME_COPYRIGHT.txt"
find "$output_root/bin" -type f -exec x86_64-w64-mingw32-strip {} \;
sha256sum "$output_root"/bin/* > "$output_root/source/native-sha256.txt"
