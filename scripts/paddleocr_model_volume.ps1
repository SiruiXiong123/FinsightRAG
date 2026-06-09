$DefaultPaddleOcrVlOfflineImage = "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-nvidia-gpu-sm120-offline"
$DefaultPaddleOcrVlServerImage = "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu-sm120"
$DefaultPaddleOcrVlModelVolume = "paddleocrvl-model"
$DefaultPaddleOcrVlContainerModelDir = "/home/paddleocr/.paddlex/official_models/PaddleOCR-VL-1.6"

function Select-FirstValue {
    foreach ($Value in $args) {
        if ($null -ne $Value -and -not [string]::IsNullOrWhiteSpace([string]$Value)) {
            return [string]$Value
        }
    }
    return $null
}

function ConvertTo-BoolValue {
    param(
        [object]$Value,
        [bool]$Default = $false
    )

    if ($null -eq $Value) {
        return $Default
    }

    $Text = ([string]$Value).Trim().ToLowerInvariant()
    if ([string]::IsNullOrWhiteSpace($Text)) {
        return $Default
    }

    if (@("1", "true", "yes", "y", "on") -contains $Text) {
        return $true
    }

    if (@("0", "false", "no", "n", "off") -contains $Text) {
        return $false
    }

    return $Default
}

function Get-FlatYamlValue {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string[]]$Keys
    )

    if (!(Test-Path -LiteralPath $Path)) {
        return $null
    }

    $Lines = Get-Content -LiteralPath $Path
    foreach ($Key in $Keys) {
        $EscapedKey = [regex]::Escape($Key)
        foreach ($Line in $Lines) {
            $Trimmed = $Line.Trim()
            if (!$Trimmed -or $Trimmed.StartsWith("#")) {
                continue
            }
            if ($Trimmed -match "^$EscapedKey\s*:\s*(.*)$") {
                $Value = $Matches[1].Trim()
                if ($Value.Contains(" #")) {
                    $Value = $Value.Split(" #", 2)[0].Trim()
                }
                return $Value.Trim("'`"")
            }
        }
    }

    return $null
}

function Get-FlatYamlBool {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string[]]$Keys,
        [bool]$Default = $false
    )

    return ConvertTo-BoolValue (Get-FlatYamlValue -Path $Path -Keys $Keys) $Default
}

function Resolve-ProjectPath {
    param(
        [string]$Path,
        [string]$ProjectRoot
    )

    if ([string]::IsNullOrWhiteSpace($Path)) {
        return $null
    }

    $ExpandedPath = [Environment]::ExpandEnvironmentVariables($Path)
    if ([System.IO.Path]::IsPathRooted($ExpandedPath)) {
        return $ExpandedPath
    }

    return Join-Path $ProjectRoot $ExpandedPath
}

function Assert-DockerAvailable {
    docker version --format "{{.Server.Version}}" | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Docker is not available. Start Docker Desktop, then retry."
    }
}

function Test-PaddleOcrVlVolumeModel {
    param(
        [Parameter(Mandatory = $true)][string]$VolumeName,
        [Parameter(Mandatory = $true)][string]$ContainerModelDir,
        [Parameter(Mandatory = $true)][string]$ProbeImage
    )

    $CheckCommand = "test -f '$ContainerModelDir/model.safetensors'"
    $DockerArgs = @(
        "run",
        "--rm",
        "--entrypoint", "sh",
        "-v", "${VolumeName}:${ContainerModelDir}:ro",
        $ProbeImage,
        "-lc", $CheckCommand
    )

    docker @DockerArgs | Out-Null
    return $LASTEXITCODE -eq 0
}

function Initialize-PaddleOcrVlModelVolume {
    param(
        [string]$VolumeName = $DefaultPaddleOcrVlModelVolume,
        [string]$ContainerModelDir = $DefaultPaddleOcrVlContainerModelDir,
        [string]$SeedImage = $DefaultPaddleOcrVlOfflineImage,
        [string]$LocalModelDir = $null
    )

    Write-Host "Ensuring Docker model volume exists: $VolumeName"
    docker volume inspect $VolumeName 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) {
        docker volume create $VolumeName | Out-Null
    }

    if (Test-PaddleOcrVlVolumeModel -VolumeName $VolumeName -ContainerModelDir $ContainerModelDir -ProbeImage $SeedImage) {
        Write-Host "Model volume is already initialized."
        return
    }

    Write-Host "Initializing $VolumeName from offline image model cache..."
    $SeedCommand = "test -f '$ContainerModelDir/model.safetensors'"
    $SeedArgs = @(
        "run",
        "--rm",
        "--entrypoint", "sh",
        "-v", "${VolumeName}:${ContainerModelDir}",
        $SeedImage,
        "-lc", $SeedCommand
    )
    docker @SeedArgs | Out-Null

    if (Test-PaddleOcrVlVolumeModel -VolumeName $VolumeName -ContainerModelDir $ContainerModelDir -ProbeImage $SeedImage) {
        Write-Host "Model volume initialized from offline image."
        return
    }

    if (![string]::IsNullOrWhiteSpace($LocalModelDir) -and (Test-Path -LiteralPath $LocalModelDir)) {
        $ResolvedLocalModelDir = (Resolve-Path -LiteralPath $LocalModelDir).Path
        Write-Host "Offline image prefill did not expose the model; copying from local model dir:"
        Write-Host $ResolvedLocalModelDir

        $CopyCommand = "mkdir -p '$ContainerModelDir' && cp -a /tmp/paddleocrvl-model/. '$ContainerModelDir/' && test -f '$ContainerModelDir/model.safetensors'"
        $CopyArgs = @(
            "run",
            "--rm",
            "--entrypoint", "sh",
            "-v", "${VolumeName}:${ContainerModelDir}",
            "-v", "${ResolvedLocalModelDir}:/tmp/paddleocrvl-model:ro",
            $SeedImage,
            "-lc", $CopyCommand
        )
        docker @CopyArgs | Out-Null
    }

    if (!(Test-PaddleOcrVlVolumeModel -VolumeName $VolumeName -ContainerModelDir $ContainerModelDir -ProbeImage $SeedImage)) {
        throw "Failed to initialize $VolumeName. Expected model.safetensors under $ContainerModelDir."
    }

    Write-Host "Model volume initialized from local model directory."
}
