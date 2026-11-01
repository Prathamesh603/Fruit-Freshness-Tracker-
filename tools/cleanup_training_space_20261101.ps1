$ErrorActionPreference='Stop'
$projectRoot=[IO.Path]::GetFullPath('D:\Projects\Fruit freshness detection')
$gradleRoot=[IO.Path]::GetFullPath('D:\.gradle')
$record=Join-Path $projectRoot 'datasets\storage-cleanup-2026-11-01\cleanup.json'
$relativeTargets=@(
 'ml\data\raw\mendeley_bdd69gyhv8\Original_Image.zip',
 'ml\data\raw\mendeley_bdd69gyhv8\originals',
 'datasets\fruit13\raw\fruit16k\publisher.zip',
 'datasets\zenodo-fruit9-v1\downloads\AgricultureDataset.zip',
 'detector_training\yolov8n-fruit9-v1\data\images\train',
 'detector_training\yolov8n-fruit9-v1\data\images\val',
 'detector_training\yolov8n-fruit9-v1\data\labels\train',
 'detector_training\yolov8n-fruit9-v1\data\labels\val',
 'android\.gradle','android\.kotlin','android\app\build','android\app\.cxx',
 'android\.tools\gradle-home\caches','android\.tools\gradle-home\.tmp',
 'apps\01-fruit-lens-classic\source\android\.gradle',
 'apps\01-fruit-lens-classic\source\android\.kotlin',
 'apps\01-fruit-lens-classic\source\android\app\build',
 'apps\01-fruit-lens-classic\source\android\app\.cxx',
 'apps\01-fruit-lens-classic\editions\02-classic-plus\android\.gradle',
 'apps\01-fruit-lens-classic\editions\02-classic-plus\android\.kotlin',
 'apps\01-fruit-lens-classic\editions\02-classic-plus\android\app\build',
 'apps\01-fruit-lens-classic\editions\02-classic-plus\android\app\.cxx',
 'apps\02-fruit-lens-custom\android\.gradle',
 'apps\02-fruit-lens-custom\android\.kotlin',
 'apps\02-fruit-lens-custom\android\app\build',
 'apps\02-fruit-lens-custom\android\app\.cxx'
)
$targets=@($relativeTargets | ForEach-Object { [IO.Path]::GetFullPath((Join-Path $projectRoot $_)) })
# Explicit shared Gradle cache roots only; never settings, credentials or JDKs.
$targets+=@((Join-Path $gradleRoot 'wrapper\dists'),(Join-Path $gradleRoot 'caches'),(Join-Path $gradleRoot '.tmp'))
$before=(Get-PSDrive D).Free
$records=@()
foreach($target in $targets){
 if(-not (Test-Path -LiteralPath $target)){continue}
 $resolved=(Resolve-Path -LiteralPath $target).ProviderPath
 if(-not ($resolved.StartsWith($projectRoot+'\',[StringComparison]::OrdinalIgnoreCase) -or $resolved.StartsWith($gradleRoot+'\',[StringComparison]::OrdinalIgnoreCase))){throw "Outside allowed roots: $resolved"}
 $item=Get-Item -LiteralPath $resolved -Force
 if($item.Attributes -band [IO.FileAttributes]::ReparsePoint){throw "Linked target refused: $resolved"}
 $files=if($item.PSIsContainer){@(Get-ChildItem -LiteralPath $resolved -File -Recurse -Force)}else{@($item)}
 $records+=[pscustomobject]@{path=$resolved;bytes=[long](($files | Measure-Object Length -Sum).Sum);files=$files.Count;removed=$false}
}
@{before_free_bytes=$before;targets=$records}|ConvertTo-Json -Depth 6|Set-Content -LiteralPath $record -Encoding utf8
foreach($entry in $records){
 $resolved=(Resolve-Path -LiteralPath $entry.path).ProviderPath
 if(-not ($resolved.StartsWith($projectRoot+'\',[StringComparison]::OrdinalIgnoreCase) -or $resolved.StartsWith($gradleRoot+'\',[StringComparison]::OrdinalIgnoreCase))){throw "Deletion outside allowed roots: $resolved"}
 Remove-Item -LiteralPath $resolved -Recurse -Force
 $entry.removed=-not (Test-Path -LiteralPath $resolved)
 @{before_free_bytes=$before;after_free_bytes=(Get-PSDrive D).Free;targets=$records}|ConvertTo-Json -Depth 6|Set-Content -LiteralPath $record -Encoding utf8
 Write-Output ('Removed: '+$resolved)
}
Write-Output ('Free bytes before: '+$before+'; after: '+(Get-PSDrive D).Free)
