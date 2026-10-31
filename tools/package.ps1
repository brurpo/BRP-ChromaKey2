# Assembles a portable app folder from the build outputs:
#   target\release\{ck-desktop.exe, ck-native.exe, ck_plan_executor.dll}  (cargo + build-executor)
#   .tools\runtime\*.dll                                                    (fetch-runtime)
#   <MediaDir>\bin\*                                                         (build-media.sh)
# and writes release.json, inventory.json and assets-manifest.json (the model download list
# from assets/windows-x86_64-directml.json, with the packaged files added so the app can check
# its own install), then zips the folder as dist\BRP-ChromaKey2-<version>-windows-x64.zip.
# Models are not packaged; the app downloads them on first start.
param(
    [Parameter(Mandatory = $true)][string]$MediaDir,
    [Parameter(Mandatory = $true)][string]$Version,
    [string]$Destination,
    [string]$BuildDir
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
if (-not $BuildDir) { $BuildDir = Join-Path $root 'target\release' }
if (-not $Destination) { $Destination = Join-Path $root 'dist\BRP-ChromaKey2' }
$runtime = Join-Path $root '.tools\runtime'
New-Item -ItemType Directory -Path $Destination -Force | Out-Null
$dest = (Resolve-Path -LiteralPath $Destination).Path
$entries = [Collections.Generic.List[object]]::new()

function Add-File([string]$Source, [string]$Relative, [string]$Component) {
    if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) { throw "Missing package input: $Source" }
    $target = Join-Path $dest $Relative
    New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
    Copy-Item -LiteralPath $Source -Destination $target -Force
    $entries.Add([pscustomobject][ordered]@{
        file = $Relative.Replace('\', '/'); component = $Component
        bytes = (Get-Item -LiteralPath $target).Length
        sha256 = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
    })
}

Add-File (Join-Path $BuildDir 'ck-desktop.exe') 'BRP-Chroma.exe' 'application'
# The installed version; the in-app updater compares it with the latest GitHub release.
$releaseJson = Join-Path ([IO.Path]::GetTempPath()) "brp-release-$PID.json"
[IO.File]::WriteAllText($releaseJson, (@{ version = $Version } | ConvertTo-Json), [Text.UTF8Encoding]::new($false))
Add-File $releaseJson 'release.json' 'application'
Remove-Item -LiteralPath $releaseJson
Add-File (Join-Path $BuildDir 'ck-native.exe') 'ck-native.exe' 'application'
Add-File (Join-Path $BuildDir 'ck_plan_executor.dll') 'ck_plan_executor.dll' 'application'
$runtimeNames = 'onnxruntime.dll', 'onnxruntime_providers_shared.dll', 'DirectML.dll', 'msvcp140.dll', 'msvcp140_1.dll', 'vcruntime140.dll', 'vcruntime140_1.dll'
foreach ($name in $runtimeNames) { Add-File (Join-Path $runtime $name) $name 'inference_runtime' }
$mediaNames = 'ffmpeg.exe', 'ffprobe.exe', 'avcodec-63.dll', 'avdevice-63.dll', 'avfilter-12.dll', 'avformat-63.dll', 'avutil-61.dll', 'swresample-7.dll', 'swscale-10.dll', 'libwinpthread-1.dll'
foreach ($name in $mediaNames) { Add-File (Join-Path $MediaDir "bin\$name") "media\$name" 'media' }

Add-File (Join-Path $root 'LICENSE') 'licenses\BRP_CHROMAKEY2_LICENSE.txt' 'notices'
Add-File (Join-Path $root 'LICENSING.md') 'licenses\LICENSING.md' 'notices'
Add-File (Join-Path $root 'crates\ck-native\kernels\LICENSE-PYTORCH.txt') 'licenses\NATIVE_KERNELS_LICENSE.txt' 'notices'
Add-File (Join-Path $root 'licenses\THIRD_PARTY_RUST_NOTICES.txt') 'licenses\THIRD_PARTY_RUST_NOTICES.txt' 'notices'
foreach ($file in Get-ChildItem -LiteralPath (Join-Path $root 'licenses\fonts') -Recurse -File) {
    Add-File $file.FullName ('licenses\fonts\' + $file.Name) 'notices'
}
foreach ($file in Get-ChildItem -LiteralPath (Join-Path $root 'licenses\pinned') -Recurse -File | Where-Object Name -NE 'sources.json') {
    Add-File $file.FullName ('licenses\pinned\' + $file.Directory.Name + '\' + $file.Name) 'notices'
}
foreach ($file in Get-ChildItem -LiteralPath (Join-Path $root 'licenses\runtime') -File) {
    Add-File $file.FullName "licenses\$($file.Name)" 'notices'
}
foreach ($file in Get-ChildItem -LiteralPath (Join-Path $root 'licenses\models') -File) {
    Add-File $file.FullName "licenses\models\$($file.Name)" 'notices'
}

# Download list: the models plus every packaged file except the GUI, release.json (rewritten
# by the updater) and the two lists.
$catalog = Get-Content -LiteralPath (Join-Path $root 'assets\windows-x86_64-directml.json') -Raw | ConvertFrom-Json
$files = [Collections.Generic.List[object]]::new()
foreach ($entry in $entries) {
    if ($entry.file -in 'BRP-Chroma.exe', 'release.json') { continue }
    $group = if ($entry.component -eq 'inference_runtime') { 'runtime' } else { 'core' }
    $files.Add([pscustomobject][ordered]@{ group = $group; path = $entry.file; url = ''; size = $entry.bytes; sha256 = $entry.sha256; bundled_only = $true })
}
foreach ($model in $catalog.files) { $files.Add($model) }
$manifest = [ordered]@{ schema_version = 1; id = $catalog.id; version = $catalog.version; files = $files.ToArray() }
$manifestPath = Join-Path $dest 'assets-manifest.json'
[IO.File]::WriteAllText($manifestPath, ($manifest | ConvertTo-Json -Depth 6) + "`n", [Text.UTF8Encoding]::new($false))
$entries.Add([pscustomobject][ordered]@{
    file = 'assets-manifest.json'; component = 'asset_catalog'
    bytes = (Get-Item -LiteralPath $manifestPath).Length
    sha256 = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
})

$totals = [ordered]@{}
foreach ($component in 'application', 'inference_runtime', 'media', 'notices', 'asset_catalog') {
    $totals[$component] = [long](($entries | Where-Object component -EQ $component | Measure-Object bytes -Sum).Sum)
}
$inventory = [ordered]@{
    schema_version = 1; kind = 'native_desktop_package'; acceleration = 'directml'
    model_weights_included = $false; python_included = $false; media_included = $true
    runtime_bundled = $true; normal_downloads = 'models_only'; component_bytes = $totals; files = $entries.ToArray()
}
[IO.File]::WriteAllText((Join-Path $dest 'inventory.json'), ($inventory | ConvertTo-Json -Depth 6) + "`n", [Text.UTF8Encoding]::new($false))
Write-Output "Packaged $($entries.Count) files into $dest"

# The release zip: one top-level folder, '/'-separated entry names.
Add-Type -AssemblyName System.IO.Compression, System.IO.Compression.FileSystem
$zipPath = Join-Path (Split-Path -Parent $dest) "BRP-ChromaKey2-$Version-windows-x64.zip"
if (Test-Path -LiteralPath $zipPath) { Remove-Item -LiteralPath $zipPath }
$zip = [IO.Compression.ZipFile]::Open($zipPath, [IO.Compression.ZipArchiveMode]::Create)
try {
    foreach ($file in @($entries.file) + 'inventory.json') {
        $source = Join-Path $dest $file
        [void][IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $source, "BRP-ChromaKey2/$file", [IO.Compression.CompressionLevel]::Optimal)
    }
} finally { $zip.Dispose() }
Write-Output ("Release zip: {0} ({1:N1} MB)" -f $zipPath, ((Get-Item -LiteralPath $zipPath).Length / 1MB))
