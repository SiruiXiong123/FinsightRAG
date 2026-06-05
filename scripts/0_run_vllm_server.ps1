
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$ProjectRoot = $null,
    [string]$Config = $null,
    [string]$ServerName = "paddle-vlm-server",
    [int]$Port = 0,
    [switch]$Force,
    [switch]$SkipVolumeInit
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

$DisableVllm = Get-FlatYamlBool -Path $AppConfig -Keys @("paddleocrvl_disable_vllm", "DisableVllm") -Default $false
$DisableVllm = ConvertTo-BoolValue $env:PADDLEOCRVL_DISABLE_VLLM $DisableVllm

if ($DisableVllm -and !$Force) {
    Write-Host "paddleocrvl_disable_vllm is true; skipping vLLM server startup."
    Write-Host "Use -Force to start the vLLM server anyway."
    exit 0
}

$ConfiguredVllmConfig = Get-FlatYamlValue -Path $AppConfig -Keys @("paddleocrvl_vllm_config", "VllmConfig")
$Config = Select-FirstValue $Config $env:PADDLEOCRVL_VLLM_CONFIG (Resolve-ProjectPath $ConfiguredVllmConfig $ProjectRoot) (Join-Path $ScriptDir "0_vllm_server_config.yml")
$Config = (Resolve-Path -LiteralPath $Config).Path

$PortOverride = if ($Port -gt 0) { "$Port" } else { $null }
$PortValue = Select-FirstValue $PortOverride $env:PADDLEOCRVL_SERVER_PORT (Get-FlatYamlValue -Path $AppConfig -Keys @("paddleocrvl_server_port", "VllmPort")) "8118"
$Port = [int]$PortValue

$ServerImage = Select-FirstValue $env:PADDLEOCRVL_SERVER_IMAGE (Get-FlatYamlValue -Path $AppConfig -Keys @("paddleocrvl_server_image", "ServerImage")) $DefaultPaddleOcrVlServerImage
$OfflineImage = Select-FirstValue $env:PADDLEOCRVL_OFFLINE_IMAGE (Get-FlatYamlValue -Path $AppConfig -Keys @("paddleocrvl_offline_image", "ClientImage")) $DefaultPaddleOcrVlOfflineImage
$VolumeName = Select-FirstValue $env:PADDLEOCRVL_MODEL_VOLUME (Get-FlatYamlValue -Path $AppConfig -Keys @("paddleocrvl_model_volume", "ModelVolume")) $DefaultPaddleOcrVlModelVolume
$ContainerModelDir = Select-FirstValue $env:PADDLEOCRVL_CONTAINER_MODEL_DIR (Get-FlatYamlValue -Path $AppConfig -Keys @("paddleocrvl_container_model_dir", "ModelDir")) $DefaultPaddleOcrVlContainerModelDir
$ConfiguredLocalModelDir = Get-FlatYamlValue -Path $AppConfig -Keys @("paddleocrvl_local_model_dir", "LocalModelDir")
$LocalModelDir = Select-FirstValue $env:PADDLEOCRVL_LOCAL_MODEL_DIR (Resolve-ProjectPath $ConfiguredLocalModelDir $ProjectRoot) (Join-Path $ProjectRoot "models\PaddleOCR-VL-1.6")

if (!(Test-Path -LiteralPath $Config)) {
    throw "vLLM config file not found: $Config"
}

Assert-DockerAvailable

if (!$SkipVolumeInit) {
    Initialize-PaddleOcrVlModelVolume `
        -VolumeName $VolumeName `
        -ContainerModelDir $ContainerModelDir `
        -SeedImage $OfflineImage `
        -LocalModelDir $LocalModelDir
}

Write-Host "Removing old server container if it exists: $ServerName"
$ExistingContainer = docker ps -a --format "{{.Names}}" | Where-Object { $_ -eq $ServerName }
if ($ExistingContainer) {
    docker rm -f $ServerName | Out-Null
} else {
    Write-Host "No existing container to remove."
}

Write-Host "Starting PaddleOCR-VL vLLM server with shared model volume: $VolumeName"

$DockerArgs = @(
    "run",
    "-d",
    "--name", $ServerName,
    "--user", "root",
    "--gpus", "all",
    "-p", "${Port}:${Port}",
    "-e", "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True",
    "-v", "${Config}:/tmp/vllm_config.yml:ro",
    "-v", "${VolumeName}:${ContainerModelDir}:ro",
    $ServerImage,
    "paddleocr", "genai_server",
    "--model_name", "PaddleOCR-VL-1.6-0.9B",
    "--model_dir", $ContainerModelDir,
    "--host", "0.0.0.0",
    "--port", "$Port",
    "--backend", "vllm",
    "--backend_config", "/tmp/vllm_config.yml"
)

docker @DockerArgs

Write-Host ""
Write-Host "Server container started."
Write-Host "Container name: $ServerName"
Write-Host "Model volume: $VolumeName -> $ContainerModelDir (read-only)"
Write-Host "API endpoint: http://localhost:$Port/v1"
Write-Host ""
Write-Host "Waiting for vLLM API to become ready..."

$ModelsUrl = "http://localhost:$Port/v1/models"
$Deadline = (Get-Date).AddSeconds(120)
$Response = $null

while ((Get-Date) -lt $Deadline) {
    try {
        $Response = Invoke-WebRequest -UseBasicParsing $ModelsUrl
        if ($Response.StatusCode -eq 200) {
            break
        }
    } catch {
        Start-Sleep -Seconds 5
    }
}

if ($Response -and $Response.StatusCode -eq 200) {
    Write-Host "vLLM server is reachable."
    Write-Host $Response.Content
} else {
    Write-Host "Server container started, but API is not ready yet."
    Write-Host "You can check logs with:"
    Write-Host "docker logs -f $ServerName"
    Write-Host ""
    Write-Host "Or check manually with:"
    Write-Host "Invoke-WebRequest -UseBasicParsing $ModelsUrl"
}
