# Downloads the redistributable inference runtime into .tools/runtime and checks every DLL
# against the hashes the app was tested with:
#   ONNX Runtime 1.23.0 (DirectML build)  - NuGet Microsoft.ML.OnnxRuntime.DirectML
#   DirectML 1.15.4                       - NuGet Microsoft.AI.DirectML (also provides headers)
#   Visual C++ 2015-2022 runtime 14.39    - Microsoft.VCLibs desktop package
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$out = Join-Path $root '.tools\runtime'
$work = Join-Path $root '.tools\downloads'
New-Item -ItemType Directory -Path $out, $work -Force | Out-Null

$expected = @{
    'onnxruntime.dll'                  = 'f5131591edac6b0a8090d0e329040a49319d7a689cb5b465235fbf7030fa8027'
    'onnxruntime_providers_shared.dll' = '3b27e1417d12b73a6a34d80414c083e359e092d2f0ce572d7e67be8cdbe9e825'
    'DirectML.dll'                     = '9c9e6d822561c6c41b90e6994b3e8857cf1d66dbfb1e0c4c799c7c89b4e92da1'
    'msvcp140.dll'                     = '6fdc2c6fc1362c317ed65ae331dda329d4d03826feb8d79d635eaf8ba49764bc'
    'msvcp140_1.dll'                   = '6284349a2262b2329aa75aa8559980f9b2775dc5ec3a9e9ecdf55648ac7f625a'
    'vcruntime140.dll'                 = 'd5a99b38807c3df803876ce5f1e71dc43cd7ac2c882582387e550941bc99cd00'
    'vcruntime140_1.dll'               = '68f948ea61695f85ff24d249077e312abb23267b3adc43679d35d781e9aa0244'
}

function Get-Package([string]$Name, [string]$Url, [string]$Sha256) {
    $zip = Join-Path $work "$Name.zip"
    $dir = Join-Path $work $Name
    if (-not (Test-Path -LiteralPath $dir)) {
        Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $zip
        if ($Sha256 -and (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Sha256) {
            throw "$Name download hash mismatch."
        }
        Expand-Archive -LiteralPath $zip -DestinationPath $dir -Force
        Remove-Item -LiteralPath $zip
    }
    $dir
}

$ort = Get-Package 'onnxruntime-directml-1.23.0' 'https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.DirectML/1.23.0' ''
$dml = Get-Package 'directml-1.15.4' 'https://www.nuget.org/api/v2/package/Microsoft.AI.DirectML/1.15.4' ''
$vc = Get-Package 'vclibs-14.0.33321' 'https://download.microsoft.com/download/4/7/c/47c6134b-d61f-4024-83bd-b9c9ea951c25/Microsoft.VCLibs.x64.14.00.Desktop.appx' 'b56a9101f706f9d95f815f5b7fa6efbac972e86573d378b96a07cff5540c5961'

Copy-Item (Join-Path $ort 'runtimes\win-x64\native\onnxruntime.dll'), (Join-Path $ort 'runtimes\win-x64\native\onnxruntime_providers_shared.dll') $out -Force
Copy-Item (Join-Path $dml 'bin\x64-win\DirectML.dll') $out -Force
foreach ($name in 'msvcp140.dll', 'msvcp140_1.dll', 'vcruntime140.dll', 'vcruntime140_1.dll') {
    $file = Get-ChildItem -LiteralPath $vc -Recurse -File -Filter $name | Select-Object -First 1
    if (-not $file) { throw "$name not found in the VCLibs package." }
    Copy-Item -LiteralPath $file.FullName -Destination $out -Force
}
foreach ($name in $expected.Keys) {
    $hash = (Get-FileHash -LiteralPath (Join-Path $out $name) -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($hash -ne $expected[$name]) { throw "$name does not match the tested build ($hash)." }
}
$headers = Join-Path $root '.tools\directml-include'
New-Item -ItemType Directory -Path $headers -Force | Out-Null
Copy-Item -Path (Join-Path $dml 'include\*') -Destination $headers -Recurse -Force
Write-Output "Runtime ready in $out (DirectML headers in $headers)."
