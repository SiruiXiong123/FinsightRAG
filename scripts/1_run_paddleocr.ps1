[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$ProjectRoot = $null,
    [string]$ServerUrlForHost = $null,
    [string]$ServerUrlForContainer = $null,
    [string]$ReportPath = $null,
    [switch]$SkipServerCheck,
    [switch]$NoAutoStartVllm,
    [switch]$DisableVllm,
    [switch]$UseVllm,
    [switch]$NoTiming,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$BatchArgs
)

$ErrorActionPreference = "Stop"

$ScriptDir = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) {
    Split-Path -Parent $MyInvocation.MyCommand.Path
} else {
    $PSScriptRoot
}

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
}

. (Join-Path $ScriptDir "0_model_volume.ps1")

$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$AppConfig = Join-Path $ProjectRoot "config.yaml"
$BatchModule = "src.ocr_batch"

if ($DisableVllm -and $UseVllm) {
    throw "Use only one of -DisableVllm or -UseVllm."
}

if (!$NoTiming) {
    if ([string]::IsNullOrWhiteSpace($ReportPath)) {
        $ReportDir = Join-Path $ProjectRoot "runs\ocr_timing"
        New-Item -ItemType Directory -Force -Path $ReportDir | Out-Null
        $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $ReportPath = Join-Path $ReportDir "paddleocr_timing_$Stamp.json"
    } else {
        $ReportParent = Split-Path -Parent $ReportPath
        if (![string]::IsNullOrWhiteSpace($ReportParent)) {
            New-Item -ItemType Directory -Force -Path $ReportParent | Out-Null
        }
    }
}

if ($null -eq $BatchArgs) {
    $BatchArgs = @()
}

$StartedAt = Get-Date
$Timer = [System.Diagnostics.Stopwatch]::StartNew()
$ExitCode = 1

$DisableVllmFromConfig = Get-FlatYamlBool -Path $AppConfig -Keys @("paddleocrvl_disable_vllm", "DisableVllm") -Default $false
$DisableVllmFromConfig = ConvertTo-BoolValue $env:PADDLEOCRVL_DISABLE_VLLM $DisableVllmFromConfig
$UseVllmEnabled = -not $DisableVllmFromConfig
if ($DisableVllm) {
    $UseVllmEnabled = $false
}
if ($UseVllm) {
    $UseVllmEnabled = $true
}

$AutoStartVllm = Get-FlatYamlBool -Path $AppConfig -Keys @("paddleocrvl_auto_start_vllm", "AutoStartVllm") -Default $true
$AutoStartVllm = ConvertTo-BoolValue $env:PADDLEOCRVL_AUTO_START_VLLM $AutoStartVllm
if ($NoAutoStartVllm) {
    $AutoStartVllm = $false
}

$PortValue = Select-FirstValue $env:PADDLEOCRVL_SERVER_PORT (Get-FlatYamlValue -Path $AppConfig -Keys @("paddleocrvl_server_port", "VllmPort")) "8118"
$Port = [int]$PortValue

$ClientImage = Select-FirstValue $env:PADDLEOCRVL_OFFLINE_IMAGE (Get-FlatYamlValue -Path $AppConfig -Keys @("paddleocrvl_offline_image", "ClientImage")) $DefaultPaddleOcrVlOfflineImage
$VolumeName = Select-FirstValue $env:PADDLEOCRVL_MODEL_VOLUME (Get-FlatYamlValue -Path $AppConfig -Keys @("paddleocrvl_model_volume", "ModelVolume")) $DefaultPaddleOcrVlModelVolume
$ContainerModelDir = Select-FirstValue $env:PADDLEOCRVL_CONTAINER_MODEL_DIR (Get-FlatYamlValue -Path $AppConfig -Keys @("paddleocrvl_container_model_dir", "ModelDir")) $DefaultPaddleOcrVlContainerModelDir
$ConfiguredLocalModelDir = Get-FlatYamlValue -Path $AppConfig -Keys @("paddleocrvl_local_model_dir", "LocalModelDir")
$LocalModelDir = Select-FirstValue $env:PADDLEOCRVL_LOCAL_MODEL_DIR (Resolve-ProjectPath $ConfiguredLocalModelDir $ProjectRoot) (Join-Path $ProjectRoot "models\PaddleOCR-VL-1.6") "C:\Users\37945\Desktop\project1\Ocrscript\models\PaddleOCR-VL-1.6"

$ServerUrlForHost = Select-FirstValue $ServerUrlForHost $env:OCR_VLM_HOST_MODELS_URL "http://localhost:$Port/v1/models"
$ServerUrlForContainer = Select-FirstValue $ServerUrlForContainer $env:OCR_VLM_SERVER_URL "http://host.docker.internal:$Port/v1"

if (!(Test-Path -LiteralPath $ProjectRoot)) {
    throw "Project root not found: $ProjectRoot"
}

if (!(Test-Path -LiteralPath (Join-Path $ProjectRoot "src\ocr_batch.py"))) {
    throw "src\ocr_batch.py not found under project root: $ProjectRoot"
}

Assert-DockerAvailable

Initialize-PaddleOcrVlModelVolume `
    -VolumeName $VolumeName `
    -ContainerModelDir $ContainerModelDir `
    -SeedImage $ClientImage `
    -LocalModelDir $LocalModelDir

if ($UseVllmEnabled -and !$SkipServerCheck) {
    Write-Host "Checking vLLM server..."
    try {
        Invoke-WebRequest -UseBasicParsing $ServerUrlForHost | Out-Null
        Write-Host "vLLM server is reachable."
    } catch {
        if (!$AutoStartVllm) {
            throw "vLLM server is not reachable. Please run .\scripts\0_run_vllm_server.ps1 first, or set paddleocrvl_disable_vllm: true. Checked URL: $ServerUrlForHost"
        }

        Write-Host "vLLM server is not reachable. Auto-starting it now..."
        & (Join-Path $ScriptDir "0_run_vllm_server.ps1") -ProjectRoot $ProjectRoot

        Write-Host "Checking vLLM server again..."
        Invoke-WebRequest -UseBasicParsing $ServerUrlForHost | Out-Null
        Write-Host "vLLM server is reachable."
    }
}

Write-Host "Starting PaddleOCR-VL with shared model volume: $VolumeName"

if ($UseVllmEnabled) {
    Write-Host "OCR mode: vLLM server"
    $DefaultBatchArgs = @(
        "--device", "gpu:0",
        "--vl-rec-backend", "vllm-server",
        "--vl-rec-server-url", $ServerUrlForContainer
    )
} else {
    Write-Host "OCR mode: local PaddleOCR-VL inference without vLLM server"
    $RunningServer = docker ps --filter "name=paddle-vlm-server" --format "{{.Names}}" | Where-Object { $_ -eq "paddle-vlm-server" }
    if ($RunningServer) {
        Write-Host "Warning: paddle-vlm-server is still running and may occupy GPU memory."
        Write-Host "Stop it with: docker rm -f paddle-vlm-server"
    }

    $DefaultBatchArgs = @(
        "--device", "gpu:0",
        "--vl-rec-backend", "none",
        "--skip-vlm-check", "true"
    )
}

$DockerArgs = @(
    "run",
    "--rm",
    "--gpus", "all",
    "--user", "root",
    "-e", "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True",
    "-v", "${ProjectRoot}:/workspace/work",
    "-v", "${VolumeName}:${ContainerModelDir}:ro",
    "-w", "/workspace/work",
    $ClientImage,
    "python", "-m", $BatchModule
) + $DefaultBatchArgs + $BatchArgs

docker @DockerArgs
$ExitCode = $LASTEXITCODE

$Timer.Stop()
$FinishedAt = Get-Date

Write-Host ""
Write-Host "PaddleOCR run finished."

if (!$NoTiming) {
    $Report = [ordered]@{
        started_at = $StartedAt.ToString("o")
        finished_at = $FinishedAt.ToString("o")
        elapsed_seconds = [math]::Round($Timer.Elapsed.TotalSeconds, 3)
        exit_code = $ExitCode
        project_root = $ProjectRoot
        batch_args = $BatchArgs
        vllm_mode = if ($DisableVllm) { "disabled" } elseif ($UseVllm) { "forced" } else { "config" }
        output_report = $ReportPath
    }

    $Report | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $ReportPath -Encoding UTF8
    Write-Host "PaddleOCR timing report: $ReportPath"
    Write-Host ("Elapsed seconds: {0}" -f $Report.elapsed_seconds)
}

if ($ExitCode -ne 0) {
    exit $ExitCode
}
