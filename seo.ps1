[CmdletBinding()]
param(
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)]
    [string[]] $WorkflowArguments
)

$ErrorActionPreference = "Stop"

if ($env:SEO_WORKFLOW_ARGUMENTS) {
    $WorkflowArguments = @(
        $env:SEO_WORKFLOW_ARGUMENTS -split "`n" |
            Where-Object { $_ -ne "" }
    )
    Remove-Item Env:SEO_WORKFLOW_ARGUMENTS -ErrorAction SilentlyContinue
}

$utf8 = New-Object System.Text.UTF8Encoding($false)
[Console]::InputEncoding = $utf8
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:TQDM_DISABLE = "1"
$env:TOKENIZERS_PARALLELISM = "false"
$env:TRANSFORMERS_NO_ADVISORY_WARNINGS = "1"

$projectDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExecutable = Join-Path $projectDirectory ".venv\Scripts\python.exe"
$workflowScript = Join-Path $projectDirectory "seo_workflow.py"

function ConvertTo-NativeArgument {
    param([AllowEmptyString()][string] $Value)

    if ($Value -ne "" -and $Value -notmatch '[\s"]') {
        return $Value
    }

    $builder = New-Object System.Text.StringBuilder
    [void] $builder.Append('"')
    $backslashes = 0
    foreach ($character in $Value.ToCharArray()) {
        if ($character -eq '\') {
            $backslashes++
            continue
        }
        if ($character -eq '"') {
            [void] $builder.Append(('\' * (($backslashes * 2) + 1)))
            [void] $builder.Append('"')
            $backslashes = 0
            continue
        }
        if ($backslashes) {
            [void] $builder.Append(('\' * $backslashes))
            $backslashes = 0
        }
        [void] $builder.Append($character)
    }
    if ($backslashes) {
        [void] $builder.Append(('\' * ($backslashes * 2)))
    }
    [void] $builder.Append('"')
    return $builder.ToString()
}

if (-not (Test-Path -LiteralPath $pythonExecutable -PathType Leaf)) {
    throw "Project Python executable not found: $pythonExecutable"
}
if (-not (Test-Path -LiteralPath $workflowScript -PathType Leaf)) {
    throw "SEO workflow entry point not found: $workflowScript"
}
if (-not $WorkflowArguments -or $WorkflowArguments.Count -eq 0) {
    throw "Missing workflow command. Example: .\seo.ps1 status --job <job>"
}

Push-Location -LiteralPath $projectDirectory
try {
    $nativeArguments = @($workflowScript) + $WorkflowArguments
    $processInfo = New-Object System.Diagnostics.ProcessStartInfo
    $processInfo.FileName = $pythonExecutable
    $processInfo.Arguments = (
        $nativeArguments |
            ForEach-Object { ConvertTo-NativeArgument ([string] $_) }
    ) -join " "
    $processInfo.UseShellExecute = $false

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $processInfo
    [void] $process.Start()
    $process.WaitForExit()
    $workflowExitCode = $process.ExitCode
}
finally {
    Pop-Location
}

exit $workflowExitCode
