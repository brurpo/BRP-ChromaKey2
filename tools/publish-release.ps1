<#
.SYNOPSIS
  Publish an in-app update release of BRP CHROMA to GitHub (development/release tooling only).

.DESCRIPTION
  Builds brp-chroma-update.json (schema 1) from an app folder: every file listed in inventory.json,
  plus inventory.json itself and, with -IncludeModels, every file under the given model folders.
  Each entry has its relative path, size, sha256 and download URL. Files whose sha256 matches the
  previous release keep that release's URL while it still exists; only new or changed files are
  uploaded. Every manifest lists the complete file set, so any installed version updates straight
  to the newest. The release is
  created as a draft, all assets are uploaded, then it is published, so the app never sees a
  half-uploaded release. The app (ck-desktop update_ui.rs) downloads only files that differ locally.

  The token is read from $env:GITHUB_TOKEN (repo scope / contents:write) and never printed.
  -DryRun writes the manifest next to the app folder and uploads nothing.

.EXAMPLE
  $env:GITHUB_TOKEN = '...'; .\tools\publish-release.ps1 -AppDir D:\BRP_CHROMA_2\app -Version 2026.11.01.1 -NotesFile notes.md
.EXAMPLE
  .\tools\publish-release.ps1 -AppDir D:\BRP_CHROMA_2\app-native-test -Version 2026.11.01.1 -IncludeModels models/sam3.1,models/translate -DryRun
#>
param(
    [Parameter(Mandatory = $true)][string]$AppDir,
    [Parameter(Mandatory = $true)][string]$Version,
    [string]$Repository = 'brurpo/BRP-ChromaKey2',
    [string]$Notes = '',
    [string]$NotesFile = '',
    [string[]]$IncludeModels = @(),
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$AppDir = (Resolve-Path $AppDir).Path
$tag = "v$Version"
if ($NotesFile) { $Notes = Get-Content -Raw -Encoding UTF8 $NotesFile }

function Get-Sha256([string]$Path) { (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLower() }
function Get-AssetName([string]$Relative) { $Relative.Replace('/', '__') }

# Files of the release (relative, '/'-separated). release.json is written by the updater itself.
$relative = New-Object System.Collections.Generic.List[string]
$inventory = Get-Content -Raw "$AppDir\inventory.json" | ConvertFrom-Json
foreach ($f in $inventory.files) { $relative.Add(($f.file -replace '\\', '/')) }
if (-not $relative.Contains('inventory.json')) { $relative.Add('inventory.json') }
foreach ($folder in $IncludeModels) {
    $root = Join-Path $AppDir $folder
    if (-not (Test-Path $root)) { throw "Model folder not found: $root" }
    Get-ChildItem $root -Recurse -File | ForEach-Object {
        $relative.Add($_.FullName.Substring($AppDir.Length + 1).Replace('\', '/'))
    }
}
$relative = $relative | Where-Object { $_ -ne 'release.json' } | Sort-Object -Unique

$headers = @{ 'Accept' = 'application/vnd.github+json'; 'User-Agent' = 'brp-chroma-publish'; 'X-GitHub-Api-Version' = '2022-11-28' }
if (-not $DryRun) {
    if (-not $env:GITHUB_TOKEN) { throw 'Set $env:GITHUB_TOKEN (contents:write on the repository).' }
    $headers['Authorization'] = "Bearer $($env:GITHUB_TOKEN)"
}

# Previous release manifest: unchanged files keep their URL.
$previous = @{}
try {
    $latest = Invoke-RestMethod -Headers $headers -Uri "https://api.github.com/repos/$Repository/releases/latest"
    $asset = $latest.assets | Where-Object { $_.name -eq 'brp-chroma-update.json' } | Select-Object -First 1
    if ($asset) {
        $old = Invoke-RestMethod -Headers @{ 'User-Agent' = 'brp-chroma-publish' } -Uri $asset.browser_download_url
        foreach ($f in $old.files) { $previous["$($f.path)|$($f.sha256)"] = $f.url }
        Write-Host "Previous release $($latest.tag_name): $($old.files.Count) files"
    }
} catch {
    Write-Host "No previous release found ($($_.Exception.Message.Split([Environment]::NewLine)[0]))."
}

$entries = @()
$uploads = @()
foreach ($rel in $relative) {
    $path = Join-Path $AppDir ($rel.Replace('/', '\'))
    if (-not (Test-Path -LiteralPath $path)) { throw "Listed file missing: $rel" }
    $sha = Get-Sha256 $path
    $size = (Get-Item -LiteralPath $path).Length
    $url = $previous["$rel|$sha"]
    # Any installed version downloads from these URLs, so reuse an old asset only while it still exists.
    if ($url) {
        try { Invoke-WebRequest -Method Head -UseBasicParsing -Uri $url -Headers @{ 'User-Agent' = 'brp-chroma-publish' } | Out-Null }
        catch { Write-Host "Re-uploading $rel (old asset missing)"; $url = $null }
    }
    if (-not $url) {
        $url = "https://github.com/$Repository/releases/download/$tag/$(Get-AssetName $rel)"
        $uploads += [pscustomobject]@{ Path = $path; Name = (Get-AssetName $rel); Size = $size }
    }
    $entries += [ordered]@{ path = $rel; size = $size; sha256 = $sha; url = $url }
}
$manifest = [ordered]@{ schema_version = 1; version = $Version; notes = $Notes; files = $entries }
$manifestPath = Join-Path (Split-Path $AppDir -Parent) "brp-chroma-update-$Version.json"
[IO.File]::WriteAllText($manifestPath, ($manifest | ConvertTo-Json -Depth 5), (New-Object Text.UTF8Encoding $false))
$bytes = ($uploads | Measure-Object Size -Sum).Sum
Write-Host ("{0} files in release, {1} to upload ({2:N1} MB). Manifest: {3}" -f $entries.Count, $uploads.Count, ($bytes / 1MB), $manifestPath)
if ($DryRun) { Write-Host 'Dry run: nothing uploaded.'; return }
# The packaged app records its own version (the updater rewrites it after each update).
[IO.File]::WriteAllText((Join-Path $AppDir 'release.json'), (@{ version = $Version } | ConvertTo-Json), (New-Object Text.UTF8Encoding $false))

$body = @{ tag_name = $tag; name = "BRP CHROMA $Version"; body = $Notes; draft = $true } | ConvertTo-Json
$release = Invoke-RestMethod -Method Post -Headers $headers -Uri "https://api.github.com/repos/$Repository/releases" -Body $body -ContentType 'application/json'
$uploadBase = $release.upload_url -replace '\{.*\}$', ''
foreach ($u in ($uploads + [pscustomobject]@{ Path = $manifestPath; Name = 'brp-chroma-update.json'; Size = (Get-Item $manifestPath).Length })) {
    Write-Host ("Uploading {0} ({1:N1} MB)" -f $u.Name, ($u.Size / 1MB))
    Invoke-RestMethod -Method Post -Headers $headers -Uri "$($uploadBase)?name=$([uri]::EscapeDataString($u.Name))" -InFile $u.Path -ContentType 'application/octet-stream' | Out-Null
}
Invoke-RestMethod -Method Patch -Headers $headers -Uri "https://api.github.com/repos/$Repository/releases/$($release.id)" -Body (@{ draft = $false } | ConvertTo-Json) -ContentType 'application/json' | Out-Null
Write-Host "Published $tag."
