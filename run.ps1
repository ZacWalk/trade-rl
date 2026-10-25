param(
    [switch]$Release,
    [string]$Db,
    [string]$Series,
    [string]$Run,
    [string]$SmokeTest
)
$ErrorActionPreference = 'Stop'
$oldPath = $env:PATH
Push-Location $PSScriptRoot
try {
    $upstream = Join-Path $HOME '.cargo\bin'
    if (Test-Path (Join-Path $upstream 'cargo.exe')) {
        $env:PATH = "$upstream;$env:PATH"
    }
    $arguments = @('+stable', 'run', '-p', 'trl-app')
    if ($Release) { $arguments += '--release' }
    $arguments += '--'
    if ($Db) { $arguments += @('--db', $Db) }
    if ($Series) { $arguments += @('--series', $Series) }
    if ($Run) { $arguments += @('--run', $Run) }
    if ($SmokeTest) { $arguments += @('--smoke-test', $SmokeTest) }
    & cargo @arguments
    if ($LASTEXITCODE -ne 0) { throw "trade-rl exited with code $LASTEXITCODE" }
}
finally {
    Pop-Location
    $env:PATH = $oldPath
}
