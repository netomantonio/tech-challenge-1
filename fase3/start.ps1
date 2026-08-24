param(
    [int]$Port = 8010,
    [string]$HostAddress = "127.0.0.1",
    [string[]]$AllowedOrigin = @("https://assistente-protocolos-fase3.pages.dev"),
    [switch]$AllowRemoteTraining,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$DefaultPython = Join-Path $ProjectRoot ".venv-fase3\Scripts\python.exe"
$PythonPath = if ($env:FASE3_PYTHON) { $env:FASE3_PYTHON } else { $DefaultPython }
$CachePointer = Join-Path $ProjectRoot ".cache\fase3-hf-home.txt"

if (-not (Test-Path -LiteralPath $PythonPath)) {
    throw "Ambiente da Fase 3 nao encontrado. Execute 'npm run fase3:setup' primeiro."
}

& $PythonPath -c "import fastapi, langgraph, peft, torch, transformers"
if ($LASTEXITCODE -ne 0) {
    throw "Dependencias incompletas. Execute 'npm run fase3:setup' primeiro."
}

$env:PYTHONPATH = $ProjectRoot
$env:FASE3_LLM_BACKEND = "local"
$env:HF_HUB_OFFLINE = "1"
if ((-not $env:HF_HOME) -and (Test-Path -LiteralPath $CachePointer)) {
    $env:HF_HOME = (Get-Content -LiteralPath $CachePointer -Raw).Trim()
}

& $PythonPath -m fase3.manage doctor --require-runtime
if ($LASTEXITCODE -ne 0) { throw "Modelo ou adapter promovido indisponivel." }

$LaunchArgs = @("-m", "fase3", "--backend", "local", "--host", $HostAddress, "--port", $Port)
foreach ($Origin in $AllowedOrigin) { $LaunchArgs += @("--allowed-origin", $Origin) }
if ($AllowRemoteTraining) { $LaunchArgs += "--allow-remote-training" }
if ($NoBrowser -or $HostAddress -eq "0.0.0.0") { $LaunchArgs += "--no-open-browser" }

Write-Host "Iniciando backend da Fase 3 em http://127.0.0.1:$Port"
if ($HostAddress -eq "0.0.0.0") { Write-Host "Acesso pela rede local habilitado; confira os IPs exibidos abaixo." }
if ($AllowRemoteTraining) { Write-Warning "Treinamento remoto sem autenticacao esta ATIVADO." }
Write-Host "Pressione Ctrl+C para encerrar."
& $PythonPath @LaunchArgs
exit $LASTEXITCODE
