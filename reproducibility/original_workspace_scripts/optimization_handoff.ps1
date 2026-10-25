$ErrorActionPreference = 'Stop'
$project = 'C:\Users\ROG\Documents\Codex\2026-10-24\files-pasted-by-the-user-ai6103\outputs\AI6103_Assignment'
$stateFile = Join-Path $project 'results\continuation.json'
$runState = Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json
if ($runState.current_experiment -ne 'cosine' -or $runState.status -ne 'training') { throw 'Unexpected training state; no process stopped.' }
$allProcesses = @(Get-CimInstance Win32_Process)
$watcher = $allProcesses | Where-Object ProcessId -EQ $runState.watcher_pid
$training = $allProcesses | Where-Object ProcessId -EQ $runState.child_pid
if ($watcher.CommandLine -notmatch 'work\\continue_training.py' -or $training.CommandLine -notmatch 'src.experiments --experiment cosine') { throw 'Process identity mismatch; no process stopped.' }
$proof = Get-Content -LiteralPath (Join-Path $project 'results\performance_optimization\loader_equivalence.json') -Raw | ConvertFrom-Json
if ($proof.status -ne 'passed') { throw 'Loader equivalence is required before handoff.' }
$statusFile = Join-Path $project 'results\cosine\status.json'
$initialEpoch = (Get-Content -LiteralPath $statusFile -Raw | ConvertFrom-Json).completed_epochs
# Stop only the sequencer first, leaving its already-running child to finish a full epoch.
Stop-Process -Id $runState.watcher_pid
$deadline = [DateTimeOffset]::Now.AddMinutes(3)
do {
    Start-Sleep -Milliseconds 250
    $observed = Get-Content -LiteralPath $statusFile -Raw | ConvertFrom-Json
    if ($observed.status -ne 'running') { throw 'Training stopped unexpectedly before handoff.' }
    if ([DateTimeOffset]::Now -gt $deadline) { throw 'No completed epoch within handoff deadline.' }
} while ($observed.completed_epochs -le $initialEpoch)
# last.pt, CSV, history and status have all been atomically committed at this point.
Stop-Process -Id $runState.child_pid
$remaining = @(Get-CimInstance Win32_Process)
$parentIds = @([uint32]$runState.child_pid)
$workerIds = @()
do {
    $children = @($remaining | Where-Object { $_.ParentProcessId -in $parentIds -and $_.ProcessId -notin $workerIds })
    $parentIds = @($children | ForEach-Object ProcessId)
    $workerIds += $parentIds
} while ($parentIds.Count -gt 0)
if ($workerIds.Count -gt 0) { Stop-Process -Id $workerIds -ErrorAction SilentlyContinue }
if (Get-Process -Id $runState.child_pid,$runState.watcher_pid -ErrorAction SilentlyContinue) { throw 'Old training/sequencer still exists.' }
$backup = Join-Path $project ('results\performance_optimization\backup_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
New-Item -ItemType Directory -Path $backup | Out-Null
Copy-Item -LiteralPath (Join-Path $project 'results\cosine') -Destination (Join-Path $backup 'cosine_results') -Recurse
Copy-Item -LiteralPath (Join-Path $project 'checkpoints\cosine') -Destination (Join-Path $backup 'cosine_checkpoints') -Recurse
Copy-Item -LiteralPath (Join-Path $project 'src') -Destination (Join-Path $backup 'src') -Recurse
Copy-Item -LiteralPath (Join-Path $project 'configs') -Destination (Join-Path $backup 'configs') -Recurse
Copy-Item -LiteralPath $stateFile -Destination (Join-Path $backup 'continuation.before.json')
$immutableFiles = foreach ($experiment in $runState.completed_experiments) {
    Get-ChildItem -LiteralPath (Join-Path $project ('results\' + $experiment)),(Join-Path $project ('checkpoints\' + $experiment)) -File -Recurse
}
$immutable = foreach ($file in $immutableFiles) {
    @{path=$file.FullName; sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}
}
$immutable | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $backup 'completed_files.json') -Encoding utf8
$removedLocks = @()
foreach ($lockName in '.continuation.lock','.cosine.lock') {
    $lockPath = Join-Path (Join-Path $project 'results') $lockName
    if (Test-Path -LiteralPath $lockPath) {
        Copy-Item -LiteralPath $lockPath -Destination (Join-Path $backup $lockName)
        Remove-Item -LiteralPath $lockPath
        $removedLocks += $lockName
    }
}
$handoff = @{status='benchmarking'; backup=$backup; completed_epoch=$observed.completed_epochs; stopped_training_pid=$runState.child_pid; stopped_watcher_pid=$runState.watcher_pid; stopped_worker_pids=$workerIds; removed_verified_stale_locks=$removedLocks; checkpoint=(Join-Path $backup 'cosine_checkpoints\last.pt'); updated_at=[DateTimeOffset]::Now.ToString('o')}
$handoff | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $project 'results\performance_optimization\control.json') -Encoding utf8
$runState.status = 'optimization_handoff'
$runState.completed_epochs = $observed.completed_epochs
$runState.child_pid = $null
$runState.watcher_pid = $null
$runState.updated_at = [DateTimeOffset]::Now.ToString('o')
$runState | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $stateFile -Encoding utf8
$handoff | ConvertTo-Json -Depth 5
