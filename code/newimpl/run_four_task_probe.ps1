param(
    [string]$ModelName = "roberta-base",
    [string]$ResultsRoot = "newimpl\results\nlu_probe",
    [int]$TopK = 8,
    [switch]$Quick
)

$ErrorActionPreference = "Stop"
$tasks = @("sst2", "mrpc", "rte", "cola")
$oneTaskScript = Join-Path $PSScriptRoot "run_one_task_probe.ps1"

foreach ($task in $tasks) {
    $cmd = @(
        "-TaskName", $task,
        "-ModelName", $ModelName,
        "-ResultsRoot", $ResultsRoot,
        "-TopK", "$TopK"
    )
    if ($Quick) {
        $cmd += "-Quick"
    }
    & $oneTaskScript @cmd
}

python newimpl\plot_probe_results.py --results_root $ResultsRoot

Write-Host "All probe tasks complete. Results root: $ResultsRoot" -ForegroundColor Cyan
Write-Host "Combined summary: $ResultsRoot\probe_summary_table.csv" -ForegroundColor Cyan
