$Host.UI.RawUI.WindowTitle = 'Fruit Lens - Live YOLOv8 Training'
$experiment = $PSScriptRoot
$results = Join-Path $experiment 'runs\train\results.csv'
$log = Join-Path $experiment 'reports\train_console.log'
$pipeline = Join-Path $experiment 'reports\pipeline_progress.json'
while ($true) {
    Clear-Host
    Write-Host 'FRUIT LENS - LIVE YOLOv8n TRAINING' -ForegroundColor Green
    Write-Host 'Refreshes every 3 seconds. Closing this window does NOT stop training.'
    Write-Host "Full log: $log"
    Write-Host ''
    if (Test-Path -LiteralPath $results) {
        try {
            $history = @(Import-Csv -LiteralPath $results | ForEach-Object {
                $entry = $_
                $clean = [ordered]@{}
                foreach ($property in $entry.PSObject.Properties) {
                    $clean[$property.Name.Trim()] = $property.Value.Trim()
                }
                [pscustomobject]$clean
            })
            $history | Select-Object -Last 12 | Select-Object `
                @{Name='Epoch';Expression={$_.epoch}}, `
                @{Name='Box loss';Expression={$_.'train/box_loss'}}, `
                @{Name='Class loss';Expression={$_.'train/cls_loss'}}, `
                @{Name='DFL loss';Expression={$_.'train/dfl_loss'}}, `
                @{Name='Val box loss';Expression={$_.'val/box_loss'}}, `
                @{Name='mAP50';Expression={$_.'metrics/mAP50(B)'}}, `
                @{Name='mAP50-95';Expression={$_.'metrics/mAP50-95(B)'}} |
                Format-Table -AutoSize | Out-Host
        } catch { Write-Host 'Updating metrics...' }
    }
    if (Test-Path -LiteralPath $pipeline) {
        try {
            $stage = Get-Content -LiteralPath $pipeline -Raw | ConvertFrom-Json
            Write-Host "Pipeline stage: $($stage.stage)" -ForegroundColor Cyan
            if ($stage.completed) { Write-Host 'Training, evaluation and verification COMPLETE.' -ForegroundColor Green }
        } catch { }
    } else { Write-Host 'Training active; held-out evaluation follows automatically.' -ForegroundColor Cyan }
    Write-Host ''
    Write-Host 'Latest training output:' -ForegroundColor Yellow
    if (Test-Path -LiteralPath $log) { Get-Content -LiteralPath $log -Tail 4 | Out-Host }
    Start-Sleep -Seconds 3
}
