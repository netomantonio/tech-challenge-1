param(
    [string]$PythonVersion = "3.11"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvPath = Join-Path $ProjectRoot ".venv-fase3"
$PythonPath = Join-Path $VenvPath "Scripts\python.exe"
$CachePointer = Join-Path $ProjectRoot ".cache\fase3-hf-home.txt"
$ModelRelativePath = "hub\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots"

function Test-QwenCache([string]$CachePath) {
    if (-not $CachePath) { return $false }
    $Snapshots = Join-Path $CachePath $ModelRelativePath
    if (-not (Test-Path -LiteralPath $Snapshots)) { return $false }
    return [bool](
        Get-ChildItem -LiteralPath $Snapshots -Recurse -Filter "model.safetensors" -ErrorAction SilentlyContinue |
            Where-Object { Test-Path -LiteralPath $_.FullName } |
            Select-Object -First 1
    )
}

if ($env:FASE3_HF_HOME) {
    $HfHome = $env:FASE3_HF_HOME
} elseif ($env:HF_HOME) {
    $HfHome = $env:HF_HOME
} elseif (Test-Path -LiteralPath $CachePointer) {
    $HfHome = (Get-Content -LiteralPath $CachePointer -Raw).Trim()
} else {
    $CacheCandidates = @(
        (Join-Path $HOME ".cache\huggingface"),
        (Join-Path $ProjectRoot ".cache\huggingface")
    )
    foreach ($Drive in Get-PSDrive -PSProvider FileSystem) {
        $CacheCandidates += Join-Path $Drive.Root "hf-cache-fase3"
    }
    $HfHome = $CacheCandidates | Where-Object { Test-QwenCache $_ } | Select-Object -First 1
    if (-not $HfHome) {
        $HfHome = Join-Path $ProjectRoot ".cache\huggingface"
    }
}

Write-Host "Preparando ambiente da Fase 3 em $VenvPath"

$Uv = Get-Command uv -ErrorAction SilentlyContinue
if ($Uv) {
    if (-not (Test-Path -LiteralPath $PythonPath)) {
        & $Uv.Source venv --python $PythonVersion $VenvPath
        if ($LASTEXITCODE -ne 0) { throw "Falha ao criar o ambiente virtual." }
    }
    & $Uv.Source pip install `
        --python $PythonPath `
        --torch-backend auto `
        -r (Join-Path $ProjectRoot "requirements.txt") `
        -r (Join-Path $ProjectRoot "requirements-fase3.txt")
    if ($LASTEXITCODE -ne 0) { throw "Falha ao instalar as dependencias da Fase 3." }
} else {
    if (-not (Test-Path -LiteralPath $PythonPath)) {
        py "-$PythonVersion" -m venv $VenvPath
        if ($LASTEXITCODE -ne 0) { throw "Falha ao criar o ambiente virtual." }
    }
    & $PythonPath -m pip install `
        -r (Join-Path $ProjectRoot "requirements.txt") `
        -r (Join-Path $ProjectRoot "requirements-fase3.txt")
    if ($LASTEXITCODE -ne 0) { throw "Falha ao instalar as dependencias da Fase 3." }
}

& $PythonPath -c "import fastapi, langgraph, peft, torch, transformers; print(f'Ambiente pronto | torch={torch.__version__} | cuda={torch.cuda.is_available()}')"
if ($LASTEXITCODE -ne 0) { throw "O ambiente foi criado, mas a validacao das dependencias falhou." }

New-Item -ItemType Directory -Path (Split-Path $CachePointer) -Force | Out-Null
Set-Content -LiteralPath $CachePointer -Value $HfHome -Encoding ascii
$env:HF_HOME = $HfHome
Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue

if (Test-QwenCache $HfHome) {
    Write-Host "Modelo Qwen encontrado no cache: $HfHome"
} else {
    Write-Host "Baixando o modelo base Qwen uma unica vez para: $HfHome"
    & $PythonPath -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct')"
    if ($LASTEXITCODE -ne 0) { throw "Falha ao baixar o modelo base Qwen." }
    if (-not (Test-QwenCache $HfHome)) { throw "Download concluido sem os pesos esperados do Qwen." }
}

Write-Host "Setup concluido. Execute: npm run fase3"
