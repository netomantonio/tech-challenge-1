param(
    [int]$Port = 8010,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$DefaultPython = Join-Path $ProjectRoot ".venv-fase3\Scripts\python.exe"
$PythonPath = if ($env:FASE3_PYTHON) { $env:FASE3_PYTHON } else { $DefaultPython }
$Adapter = Join-Path $ProjectRoot "resultados\fase3\finetuning\qwen2.5-1.5b-v4\lora_adapter\adapter_model.safetensors"
$CachePointer = Join-Path $ProjectRoot ".cache\fase3-hf-home.txt"
$ModelRelativePath = "hub\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots"

if (-not (Test-Path -LiteralPath $PythonPath)) {
    throw "Ambiente da Fase 3 nao encontrado. Execute 'npm run fase3:setup' primeiro."
}
if (-not (Test-Path -LiteralPath $Adapter)) {
    throw "Adapter LoRA promovido nao encontrado em: $Adapter"
}

$HfHome = if ($env:FASE3_HF_HOME) {
    $env:FASE3_HF_HOME
} elseif ($env:HF_HOME) {
    $env:HF_HOME
} elseif (Test-Path -LiteralPath $CachePointer) {
    (Get-Content -LiteralPath $CachePointer -Raw).Trim()
} else {
    $null
}
if (-not $HfHome) {
    throw "Cache do modelo nao configurado. Execute 'npm run fase3:setup' primeiro."
}
$Snapshots = Join-Path $HfHome $ModelRelativePath
$ModelFile = Get-ChildItem -LiteralPath $Snapshots -Recurse -Filter "model.safetensors" -ErrorAction SilentlyContinue |
    Where-Object { Test-Path -LiteralPath $_.FullName } |
    Select-Object -First 1
if (-not $ModelFile) {
    throw "Modelo Qwen ausente no cache '$HfHome'. Execute 'npm run fase3:setup'; o download deve acontecer no setup, nunca durante uma consulta."
}

& $PythonPath -c "import fastapi, langgraph, peft, torch, transformers"
if ($LASTEXITCODE -ne 0) {
    throw "Dependencias incompletas. Execute 'npm run fase3:setup' primeiro."
}

$env:PYTHONPATH = $ProjectRoot
$env:FASE3_LLM_BACKEND = "local"
$env:HF_HOME = $HfHome
$env:HF_HUB_OFFLINE = "1"
$LaunchArgs = @("-m", "fase3", "--backend", "local", "--port", $Port)
if ($NoBrowser) { $LaunchArgs += "--no-open-browser" }

Write-Host "Iniciando Fase 3 com modelo local em http://127.0.0.1:$Port"
Write-Host "Cache Hugging Face: $HfHome (offline durante a execucao)"
Write-Host "Pressione Ctrl+C para encerrar."
& $PythonPath @LaunchArgs
exit $LASTEXITCODE
