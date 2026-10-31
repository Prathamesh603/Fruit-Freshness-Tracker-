$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath('D:\Projects\Fruit freshness detection')
$recordDir = Join-Path $projectRoot 'datasets\storage-cleanup-2026-10-31'
New-Item -ItemType Directory -Path $recordDir -Force | Out-Null
# Exact, project-contained generated data and cache paths only. Never delete SDK,
# emulator user data, signing keys, saved models, APKs, source code or reports.
$relativeTargets = @(
 'android\.gradle',
 'android\app\build',
 'android\app\.cxx',
 'android\.tools\gradle-home\caches',
 'android\.tools\gradle-home\.tmp',
 'apps\02-fruit-lens-custom\android\.gradle',
 'apps\02-fruit-lens-custom\android\app\build',
 'apps\02-fruit-lens-custom\android\app\.cxx',
 'apps\01-fruit-lens-classic\editions\02-classic-plus\android\.gradle',
 'apps\01-fruit-lens-classic\editions\02-classic-plus\android\app\build',
 'apps\01-fruit-lens-classic\editions\02-classic-plus\android\app\.cxx',
 'android\.tools\android-ndk-r28c-windows.zip',
 'android\.tools\build-tools_r36_windows.zip',
 'android\.tools\cmake-3.22.1-windows.zip',
 'android\.tools\emulator-windows_x64-16696583.zip',
 'android\.tools\gradle-8.13-bin.zip',
 'android\.tools\ncnn-20261020-android.zip',
 'android\.tools\platform-36_r02.zip',
 'android\.tools\platform-36-ext19_r01.zip',
 'android\.tools\platform-tools_r37.0.1-win.zip',
 'android\.tools\x86_64-30_r11.zip',
 'detector_training\yolov8n-fruit9-v2\data\images',
 'detector_training\yolov8n-fruit9-v2\data\labels',
 'datasets\bohni-v13\prepared',
 'ml\data\raw\mendeley_bdd69gyhv8\Augmented_Image.zip'
)
$before = (Get-PSDrive D).Free
$records = @()
foreach ($relative in $relativeTargets) {
 $absolute = [IO.Path]::GetFullPath((Join-Path $projectRoot $relative))
 if (-not $absolute.StartsWith($projectRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
  throw "Refusing path outside project: $absolute"
 }
 if (-not (Test-Path -LiteralPath $absolute)) { continue }
 $item = Get-Item -LiteralPath $absolute -Force
 if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Refusing linked target: $absolute" }
 $files = if ($item.PSIsContainer) { @(Get-ChildItem -LiteralPath $absolute -File -Recurse -Force) } else { @($item) }
 $records += [pscustomobject]@{ path=$absolute; bytes=[long](($files | Measure-Object -Property Length -Sum).Sum); files=$files.Count; removed=$false }
}
$manifest = Join-Path $recordDir 'cleanup.json'
@{created=(Get-Date -Format o);before_free_bytes=$before;targets=$records} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifest -Encoding utf8
foreach ($record in $records) {
 # Resolve and check the absolute target again immediately before deletion.
 $resolved = (Resolve-Path -LiteralPath $record.path).ProviderPath
 if (-not $resolved.StartsWith($projectRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw "Outside workspace: $resolved" }
 Remove-Item -LiteralPath $resolved -Recurse -Force
 $record.removed = -not (Test-Path -LiteralPath $resolved)
 @{created=(Get-Date -Format o);before_free_bytes=$before;after_free_bytes=(Get-PSDrive D).Free;targets=$records} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifest -Encoding utf8
 Write-Output ("Removed: " + $record.path)
}
Write-Output ("Free bytes before: " + $before + "; after: " + (Get-PSDrive D).Free)
