param(
    [string]$PythonVersion = "3.11",
    [string]$ModelAlias = "qwen2.5-1.5b",
    [switch]$SkipModel
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvPath = Join-Path $ProjectRoot ".venv-fase3"
$PythonPath = Join-Path $VenvPath "Scripts\python.exe"
$CachePointer = Join-Path $ProjectRoot ".cache\fase3-hf-home.txt"
$HfHome = if ($env:FASE3_HF_HOME) { $env:FASE3_HF_HOME } elseif ($env:HF_HOME) { $env:HF_HOME } else { Join-Path $ProjectRoot ".cache\huggingface" }

Write-Host "Preparando ambiente da Fase 3 em $VenvPath"
$Uv = Get-Command uv -ErrorAction SilentlyContinue
if ($Uv) {
    if (-not (Test-Path -LiteralPath $PythonPath)) {
        & $Uv.Source venv --python $PythonVersion $VenvPath
        if ($LASTEXITCODE -ne 0) { throw "Falha ao criar o ambiente virtual." }
    }
    & $Uv.Source pip install --python $PythonPath --torch-backend auto `
        -r (Join-Path $ProjectRoot "requirements.txt") `
        -r (Join-Path $ProjectRoot "requirements-fase3.txt")
} else {
    if (-not (Test-Path -LiteralPath $PythonPath)) {
        py "-$PythonVersion" -m venv $VenvPath
        if ($LASTEXITCODE -ne 0) { throw "Falha ao criar o ambiente virtual." }
    }
    & $PythonPath -m pip install `
        -r (Join-Path $ProjectRoot "requirements.txt") `
        -r (Join-Path $ProjectRoot "requirements-fase3.txt")
}
if ($LASTEXITCODE -ne 0) { throw "Falha ao instalar as dependencias da Fase 3." }

& $PythonPath -c "import fastapi, langgraph, peft, torch, transformers; print(f'Ambiente pronto | torch={torch.__version__} | cuda={torch.cuda.is_available()}')"
if ($LASTEXITCODE -ne 0) { throw "A validacao das dependencias falhou." }

New-Item -ItemType Directory -Path (Split-Path $CachePointer) -Force | Out-Null
Set-Content -LiteralPath $CachePointer -Value $HfHome -Encoding ascii
$env:PYTHONPATH = $ProjectRoot
$env:HF_HOME = $HfHome
Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue

if (-not $SkipModel) {
    Write-Host "Preparando modelo $ModelAlias em $HfHome"
    & $PythonPath -m fase3.manage install --alias $ModelAlias
    if ($LASTEXITCODE -ne 0) { throw "Falha ao preparar o modelo $ModelAlias." }
}

Write-Host "Setup concluido. Execute: npm run fase3"
