<#
.SYNOPSIS
  Publishes a release zip made by tools/package.ps1 as a GitHub release.

.DESCRIPTION
  The release gets one asset, BRP-ChromaKey2-<version>-windows-x64.zip. Installed copies of the
  app compare their release.json with the latest release's tag and, when it is newer, download
  this zip and replace their files, so any older version updates straight to the newest.
  The release is created as a draft, the zip is uploaded, then it is published.
  The token is read from $env:GITHUB_TOKEN (contents:write) and never printed.

.EXAMPLE
  $env:GITHUB_TOKEN = '...'; .\tools\publish-release.ps1 -Version 2026.11.1.1 -NotesFile notes.md
#>
param(
    [Parameter(Mandatory = $true)][string]$Version,
    [string]$Zip,
    [string]$Repository = 'brurpo/BRP-ChromaKey2',
    [string]$Notes = '',
    [string]$NotesFile = ''
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
if (-not $Zip) { $Zip = Join-Path $root "dist\BRP-ChromaKey2-$Version-windows-x64.zip" }
if (-not (Test-Path -LiteralPath $Zip)) { throw "Release zip not found: $Zip (run tools/package.ps1 -Version $Version)" }
if ($NotesFile) { $Notes = [IO.File]::ReadAllText((Resolve-Path -LiteralPath $NotesFile).Path) }
if (-not $env:GITHUB_TOKEN) { throw 'Set $env:GITHUB_TOKEN (contents:write on the repository).' }
$headers = @{
    Accept = 'application/vnd.github+json'; 'User-Agent' = 'brp-chromakey2-publish'
    'X-GitHub-Api-Version' = '2022-11-28'; Authorization = "Bearer $($env:GITHUB_TOKEN)"
}
# Tag the exact commit the zip was built from, and only once GitHub has it.
$commit = (git -C $root rev-parse HEAD).Trim()
git -C $root fetch --quiet origin
if (-not (git -C $root branch --remotes --contains $commit)) {
    throw "Commit $commit is not on GitHub yet. Push it (git push) before publishing."
}
$body = @{ tag_name = "v$Version"; target_commitish = $commit; name = "BRP ChromaKey2 $Version"; body = $Notes; draft = $true } | ConvertTo-Json
$release = Invoke-RestMethod -Method Post -Headers $headers -Uri "https://api.github.com/repos/$Repository/releases" -Body $body -ContentType 'application/json'
$name = [IO.Path]::GetFileName($Zip)
$upload = ($release.upload_url -replace '\{.*\}$', '') + "?name=$([uri]::EscapeDataString($name))"
Write-Host ("Uploading {0} ({1:N1} MB)" -f $name, ((Get-Item -LiteralPath $Zip).Length / 1MB))
Invoke-RestMethod -Method Post -Headers $headers -Uri $upload -InFile $Zip -ContentType 'application/zip' | Out-Null
Invoke-RestMethod -Method Patch -Headers $headers -Uri "https://api.github.com/repos/$Repository/releases/$($release.id)" -Body (@{ draft = $false } | ConvertTo-Json) -ContentType 'application/json' | Out-Null
Write-Host "Published v${Version}: $($release.html_url)"
