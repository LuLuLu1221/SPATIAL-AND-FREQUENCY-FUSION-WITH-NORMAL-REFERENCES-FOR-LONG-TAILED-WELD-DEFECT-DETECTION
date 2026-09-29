param(
    [Parameter(Mandatory = $true)] [int] $RetrievalPid
)

$run = 'D:\1\项目论文\zhwk_runs\searchdet_lvis_rare_class_pilot_20260923'
$manifest = Join-Path $run 'web_supports\support_manifest.json'
$python = 'D:\miniconda3\envs\pytorch_env\python.exe'
$script = 'D:\1\项目论文\scripts\run_searchdet_lvis_smoke.py'
$checkpoint = 'D:\1\项目论文\third_party\SearchDet\sam-hq\pretrained_checkpoint\sam_hq_vit_l.pth'
$searchdet = 'D:\1\项目论文\third_party\SearchDet'
$images = 'D:\1\项目论文\public_datasets\LVIS\images\val2017'
$stdout = Join-Path $run 'searchdet_smoke_20260923_stdout.log'
$stderr = Join-Path $run 'searchdet_smoke_20260923_stderr.log'
$status = Join-Path $run 'searchdet_smoke_launch_status.txt'

try { Wait-Process -Id $RetrievalPid -ErrorAction Stop } catch { }
if (-not (Test-Path -LiteralPath $manifest)) {
    [System.IO.File]::WriteAllText($status, "Support retrieval did not produce support_manifest.json.`r`n", [System.Text.UTF8Encoding]::new($false))
    exit 2
}
$argumentList = @(
    $script,
    '--pilot-manifest', (Join-Path $run 'pilot_manifest.json'),
    '--support-manifest', $manifest,
    '--image-root', $images,
    '--sam-checkpoint', $checkpoint,
    '--searchdet-root', $searchdet,
    '--output', (Join-Path $run 'searchdet_lvis_smoke_report.json'),
    '--limit', '10'
)
$proc = Start-Process -FilePath $python -ArgumentList $argumentList -WorkingDirectory $searchdet -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
[System.IO.File]::WriteAllText($status, "Inference PID: $($proc.Id)`r`nStarted: $([DateTime]::Now.ToString('o'))`r`n", [System.Text.UTF8Encoding]::new($false))
