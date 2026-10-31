# Installs the build toolchain into .tools/ inside this checkout (nothing system-wide):
# Rust (x86_64-pc-windows-gnu host) and LLVM-MinGW, which links the app and compiles the
# DirectML plan executor. Run once, then dot-source tools/env.ps1 in each new shell.
param([string]$RustVersion = '1.99.1')
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$tools = Join-Path $root '.tools'
New-Item -ItemType Directory -Path $tools -Force | Out-Null
$env:CARGO_HOME = Join-Path $tools 'cargo'
$env:RUSTUP_HOME = Join-Path $tools 'rustup'
# rustup reports progress on stderr, which Windows PowerShell 5.1 would turn into errors.
function Invoke-Native([string]$Exe, [string[]]$Arguments) {
    $ErrorActionPreference = 'Continue'
    & $Exe @Arguments 2>&1 | ForEach-Object { "$_" }
    if ($LASTEXITCODE -ne 0) { throw "$([IO.Path]::GetFileName($Exe)) failed ($LASTEXITCODE)." }
}

$rustup = Join-Path $env:CARGO_HOME 'bin\rustup.exe'
if (-not (Test-Path -LiteralPath $rustup)) {
    $installer = Join-Path $tools 'rustup-init.exe'
    Invoke-WebRequest -UseBasicParsing -Uri 'https://static.rust-lang.org/rustup/dist/x86_64-pc-windows-gnu/rustup-init.exe' -OutFile $installer
    Invoke-Native $installer @('-y', '--no-modify-path', '--profile', 'minimal', '--default-host', 'x86_64-pc-windows-gnu', '--default-toolchain', $RustVersion)
} else {
    Invoke-Native $rustup @('toolchain', 'install', $RustVersion, '--profile', 'minimal')
    Invoke-Native $rustup @('default', $RustVersion)
}
Invoke-Native $rustup @('component', 'add', 'rustfmt', 'clippy', '--toolchain', $RustVersion)

$llvmRelease = '20261020'
$llvmName = "llvm-mingw-$llvmRelease-ucrt-x86_64"
if (-not (Test-Path -LiteralPath (Join-Path $tools "$llvmName\bin\clang++.exe"))) {
    $archive = Join-Path $tools 'llvm-mingw.zip'
    $release = Invoke-RestMethod -Uri "https://api.github.com/repos/mstorsjo/llvm-mingw/releases/tags/$llvmRelease"
    $asset = $release.assets | Where-Object { $_.name -eq "$llvmName.zip" }
    if (-not $asset) { throw "LLVM-MinGW $llvmRelease is not available." }
    Invoke-WebRequest -UseBasicParsing -Uri $asset.browser_download_url -OutFile $archive
    if ($asset.digest -and $asset.digest.StartsWith('sha256:')) {
        if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $asset.digest.Substring(7)) {
            throw 'LLVM-MinGW archive hash mismatch.'
        }
    }
    Expand-Archive -LiteralPath $archive -DestinationPath $tools -Force
    Remove-Item -LiteralPath $archive
}
Write-Output 'Toolchain ready. Next: . .\tools\env.ps1'
