param(
    [switch]$Monitor,
    [switch]$RunGuiChild,
    [string]$ExitCodeFile
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

if ($RunGuiChild) {
    $childExitCode = 1
    try {
        if ([string]::IsNullOrWhiteSpace($ExitCodeFile)) {
            throw "ExitCodeFile is required in child mode."
        }
        $uv = Get-Command uv -ErrorAction Stop
        Push-Location $projectRoot
        try {
            & $uv.Source run --locked python "run.py" "demo\gui.pyw"
            if ($null -eq $LASTEXITCODE) {
                $childExitCode = 1
            }
            else {
                $childExitCode = [int]$LASTEXITCODE
            }
        }
        finally {
            Pop-Location
        }
    }
    catch {
        [Console]::Error.WriteLine("launcher child: " + $_.Exception.Message)
        $childExitCode = 1
    }
    finally {
        if (-not [string]::IsNullOrWhiteSpace($ExitCodeFile)) {
            try {
                Set-Content -LiteralPath $ExitCodeFile -Value ([string]$childExitCode) -Encoding Ascii -NoNewline
            }
            catch {
                [Console]::Error.WriteLine("launcher child exit-code write failed: " + $_.Exception.Message)
                $childExitCode = 1
            }
        }
    }
    exit $childExitCode
}

if (-not $Monitor) {
    try {
        $quotedScript = '"' + $PSCommandPath + '"'
        $argsLine = "-NoProfile -ExecutionPolicy Bypass -File $quotedScript -Monitor"
        Start-Process -FilePath "powershell.exe" -ArgumentList $argsLine -WindowStyle Hidden | Out-Null
        exit 0
    }
    catch {
        Write-Error ("GUI起動監視プロセスを開始できませんでした: " + $_.Exception.Message)
        exit 1
    }
}

$logBase = if ($env:LOCALAPPDATA) { $env:LOCALAPPDATA } else { $env:TEMP }
$logDir = Join-Path $logBase "packing3d"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss-fff"
$stdoutLog = Join-Path $logDir ("gui-launch-{0}-stdout.log" -f $stamp)
$stderrLog = Join-Path $logDir ("gui-launch-{0}-stderr.log" -f $stamp)
$readyFile = Join-Path $env:TEMP ("packing3d-gui-ready-{0}-{1}.txt" -f $PID, $stamp)
$exitCodeFile = Join-Path $env:TEMP ("packing3d-gui-exit-{0}-{1}.txt" -f $PID, $stamp)

function Show-LaunchFailure([string]$message) {
    try {
        Add-Type -AssemblyName PresentationFramework
        [System.Windows.MessageBox]::Show(
            $message,
            "packing3d GUI 起動エラー",
            [System.Windows.MessageBoxButton]::OK,
            [System.Windows.MessageBoxImage]::Error
        ) | Out-Null
    }
    catch {
        try {
            $shell = New-Object -ComObject WScript.Shell
            $shell.Popup($message, 0, "packing3d GUI 起動エラー", 16) | Out-Null
        }
        catch {
            # Logs remain available even if no graphical notification API works.
        }
    }
}

try {
    Get-Command uv -ErrorAction Stop | Out-Null
}
catch {
    Show-LaunchFailure("uv が見つかりません。README.md の Quick Start を確認してください。")
    exit 1
}

Remove-Item -Force -ErrorAction SilentlyContinue $readyFile
Remove-Item -Force -ErrorAction SilentlyContinue $exitCodeFile
$env:PACKING3D_GUI_READY_FILE = $readyFile

try {
    $quotedScript = '"' + $PSCommandPath + '"'
    $quotedExitCodeFile = '"' + $exitCodeFile + '"'
    $childArgs = "-NoProfile -ExecutionPolicy Bypass -File $quotedScript -RunGuiChild -ExitCodeFile $quotedExitCodeFile"
    $startParams = @{
        FilePath = "powershell.exe"
        ArgumentList = $childArgs
        WorkingDirectory = $projectRoot
        WindowStyle = "Hidden"
        RedirectStandardOutput = $stdoutLog
        RedirectStandardError = $stderrLog
        PassThru = $true
    }
    $process = Start-Process @startParams
}
catch {
    $detail = $_.Exception.Message
    Add-Content -Path $stderrLog -Value ("launcher: " + $detail)
    Show-LaunchFailure("GUI起動要求を開始できませんでした。`n`n$detail`n`nログ: $stderrLog")
    Remove-Item -Force -ErrorAction SilentlyContinue $readyFile
    Remove-Item -Force -ErrorAction SilentlyContinue $exitCodeFile
    exit 1
}

$startupReady = $false
while (-not $process.HasExited) {
    if (Test-Path $readyFile) {
        $startupReady = $true
        break
    }
    Start-Sleep -Milliseconds 100
    $process.Refresh()
}
if (Test-Path $readyFile) {
    $startupReady = $true
}

$process.WaitForExit()

$exitCode = $null
if (Test-Path $exitCodeFile) {
    try {
        $exitText = (Get-Content -LiteralPath $exitCodeFile -Raw).Trim()
        $parsedExitCode = 0
        if ([int]::TryParse($exitText, [ref]$parsedExitCode)) {
            $exitCode = $parsedExitCode
        }
    }
    catch {
        Add-Content -Path $stderrLog -Value ("launcher: failed to read child exit code: " + $_.Exception.Message)
    }
}

Remove-Item -Force -ErrorAction SilentlyContinue $readyFile
Remove-Item -Force -ErrorAction SilentlyContinue $exitCodeFile

if ($null -eq $exitCode) {
    Show-LaunchFailure(
        "GUIプロセスの終了コードを確認できませんでした。`n`n" +
        "標準エラー: $stderrLog`n標準出力: $stdoutLog"
    )
    exit 1
}

if (-not $startupReady) {
    Show-LaunchFailure(
        "GUIは起動完了を通知する前に終了しました。終了コード: $exitCode`n`n" +
        "標準エラー: $stderrLog`n標準出力: $stdoutLog"
    )
    exit 1
}

if ($exitCode -ne 0) {
    Show-LaunchFailure(
        "GUIが異常終了しました。終了コード: $exitCode`n`n" +
        "標準エラー: $stderrLog`n標準出力: $stdoutLog"
    )
    exit $exitCode
}

exit 0
