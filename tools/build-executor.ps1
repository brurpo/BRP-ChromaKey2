# Builds ck_plan_executor.dll, the raw-DirectML executor that runs CorridorKey's
# precompiled static plans (native/executor/stage1_executor.cpp). Needs the toolchain from
# tools/setup-toolchain.ps1 and the DirectML headers from tools/fetch-runtime.ps1.
param([string]$OutDir)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
if (-not $OutDir) { $OutDir = Join-Path $root 'target\release' }
$clang = Join-Path $root '.tools\llvm-mingw-20261020-ucrt-x86_64\bin\clang++.exe'
$include = Join-Path $root '.tools\directml-include'
if (-not (Test-Path -LiteralPath $clang)) { throw 'Run tools/setup-toolchain.ps1 first.' }
if (-not (Test-Path -LiteralPath (Join-Path $include 'DirectML.h'))) { throw 'Run tools/fetch-runtime.ps1 first.' }
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
$dll = Join-Path $OutDir 'ck_plan_executor.dll'
& $clang -std=c++17 -O2 -static -shared -DCK_EXECUTOR_DLL `
    -ffunction-sections -fdata-sections '-Wl,--gc-sections' -s `
    -I $include (Join-Path $root 'native\executor\stage1_executor.cpp') -o $dll
if ($LASTEXITCODE -ne 0) { throw "Executor build failed ($LASTEXITCODE)." }
Write-Output "Built $dll"
