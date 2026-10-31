# Dot-source this (". .\tools\env.ps1") to use the toolchain from tools/setup-toolchain.ps1.
# It only changes the current shell.
$root = Split-Path -Parent $PSScriptRoot
$env:CARGO_HOME = Join-Path $root '.tools\cargo'
$env:RUSTUP_HOME = Join-Path $root '.tools\rustup'
if (-not $env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR = Join-Path $root 'target' }
$env:PATH = (Join-Path $env:CARGO_HOME 'bin') + [IO.Path]::PathSeparator + $env:PATH
$rustc = Join-Path $env:CARGO_HOME 'bin\rustc.exe'
$sysroot = if (Test-Path -LiteralPath $rustc) { & $rustc --print sysroot } else { $null }
$gnu = if ($sysroot) { Join-Path $sysroot 'lib\rustlib\x86_64-pc-windows-gnu\bin\self-contained' } else { '' }
if ($gnu -and (Test-Path -LiteralPath $gnu)) { $env:PATH = $gnu + [IO.Path]::PathSeparator + $env:PATH }
$llvm = Join-Path $root '.tools\llvm-mingw-20261020-ucrt-x86_64\bin'
if (Test-Path -LiteralPath $llvm) {
    $env:PATH = $llvm + [IO.Path]::PathSeparator + $env:PATH
    $env:CC_x86_64_pc_windows_gnu = Join-Path $llvm 'clang.exe'
    # LLVM's linker handles the GUI's raw-dylib imports; the bundled GNU linker can produce
    # executables that crash before main.
    $env:CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER = Join-Path $llvm 'clang.exe'
}
# Tests that run ONNX Runtime load it from the runtime folder made by tools/fetch-runtime.ps1.
$ort = Join-Path $root '.tools\runtime\onnxruntime.dll'
if (Test-Path -LiteralPath $ort) { $env:ORT_DYLIB_PATH = $ort }
