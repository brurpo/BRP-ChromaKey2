# Writes licenses/THIRD_PARTY_RUST_NOTICES.txt: every crate in Cargo.lock for the Windows
# target with its license expression and the full license texts from its published archive.
# Identical texts are printed once. Run after a build (it reads the cargo registry sources).
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$json = & cargo metadata --format-version 1 --locked --filter-platform x86_64-pc-windows-gnu --manifest-path (Join-Path $root 'Cargo.toml')
if ($LASTEXITCODE -ne 0) { throw 'cargo metadata failed.' }
$meta = $json | ConvertFrom-Json
$members = @($meta.workspace_members)
$packages = $meta.packages | Where-Object { $members -notcontains $_.id } | Sort-Object name, version
$pattern = '^(LICEN[CS]E|COPYING|COPYRIGHT|NOTICE|UNLICENSE)([-_.].*)?$'
$texts = [ordered]@{}
$body = [Text.StringBuilder]::new()
[void]$body.AppendLine('BRP ChromaKey2 - Rust dependency licenses')
[void]$body.AppendLine("Generated from Cargo.lock for x86_64-pc-windows-gnu. $(@($packages).Count) crates.")
[void]$body.AppendLine('Crates without a license file in their archive are covered by licenses/pinned or by the')
[void]$body.AppendLine('standard text of their declared license, included below where another crate ships it.')
[void]$body.AppendLine('')
foreach ($package in $packages) {
    $dir = Split-Path -Parent $package.manifest_path
    $files = @(Get-ChildItem -LiteralPath $dir -File | Where-Object { $_.Name -match $pattern } | Sort-Object Name)
    $license = if ($package.license) { $package.license } else { 'see license file' }
    [void]$body.AppendLine("$($package.name) $($package.version) - $license")
    if ($package.repository) { [void]$body.AppendLine("  $($package.repository)") }
    foreach ($file in $files) {
        $text = [IO.File]::ReadAllText($file.FullName).Replace("`r`n", "`n").Trim()
        $hash = [BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash([Text.Encoding]::UTF8.GetBytes($text))).Replace('-', '').Substring(0, 16).ToLowerInvariant()
        if (-not $texts.Contains($hash)) { $texts[$hash] = $text }
        [void]$body.AppendLine("  $($file.Name): text [$hash]")
    }
    if (-not $files) { [void]$body.AppendLine('  (no license file in the crate archive)') }
}
[void]$body.AppendLine('')
[void]$body.AppendLine('=' * 78)
foreach ($hash in $texts.Keys) {
    [void]$body.AppendLine("[$hash]")
    [void]$body.AppendLine($texts[$hash])
    [void]$body.AppendLine('-' * 78)
}
$out = Join-Path $root 'licenses\THIRD_PARTY_RUST_NOTICES.txt'
[IO.File]::WriteAllText($out, $body.ToString().Replace("`r`n", "`n"), [Text.UTF8Encoding]::new($false))
Write-Output "Wrote $out ($(@($packages).Count) crates, $($texts.Count) distinct license texts)."
