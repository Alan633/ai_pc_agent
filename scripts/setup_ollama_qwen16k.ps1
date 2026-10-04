$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$modelFile = Join-Path $repoRoot "ollama\Modelfile.qwen3.5-9b-16k"
$modelName = "qwen3.5-9b-16k"
$baseModel = "qwen3.5:9b"
$envPath = Join-Path $repoRoot ".env"

Write-Host "[1/4] 检查 Ollama..." -ForegroundColor Cyan
$ollama = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollama) {
    throw "没有找到 ollama 命令。请先确认 Ollama 已安装并加入 PATH。"
}

Write-Host "[2/4] 检查基础模型 $baseModel ..." -ForegroundColor Cyan
$list = (& ollama list) -join "`n"
if ($LASTEXITCODE -ne 0) {
    throw "ollama list 执行失败。请确认 Ollama 服务正在运行。"
}
if ($list -notmatch [regex]::Escape($baseModel)) {
    Write-Host "本地没有 $baseModel，开始拉取..." -ForegroundColor Yellow
    & ollama pull $baseModel
    if ($LASTEXITCODE -ne 0) {
        throw "拉取 $baseModel 失败。"
    }
}

Write-Host "[3/4] 创建 16K 上下文模型 $modelName ..." -ForegroundColor Cyan
& ollama create $modelName -f $modelFile
if ($LASTEXITCODE -ne 0) {
    throw "创建 $modelName 失败。"
}

Write-Host "[4/4] 更新 .env 模型名称..." -ForegroundColor Cyan
if (Test-Path $envPath) {
    $backup = "$envPath.bak-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
    Copy-Item $envPath $backup
    Write-Host "已备份 .env -> $backup" -ForegroundColor DarkGray

    $text = Get-Content $envPath -Raw -Encoding UTF8
    if ($text -match "(?m)^OPENAI_MODEL=.*$") {
        $text = [regex]::Replace($text, "(?m)^OPENAI_MODEL=.*$", "OPENAI_MODEL=$modelName")
    } else {
        $text = $text.TrimEnd() + "`r`nOPENAI_MODEL=$modelName`r`n"
    }

    if ($text -match "(?m)^GAME_VISION_MODEL=.*$") {
        $text = [regex]::Replace($text, "(?m)^GAME_VISION_MODEL=.*$", "GAME_VISION_MODEL=$modelName")
    } else {
        $text = $text.TrimEnd() + "`r`nGAME_VISION_MODEL=$modelName`r`n"
    }

    Set-Content -Path $envPath -Value $text -Encoding UTF8
    Write-Host ".env 已更新为 $modelName" -ForegroundColor Green
} else {
    Write-Host "未找到 .env；请手动设置：" -ForegroundColor Yellow
    Write-Host "OPENAI_MODEL=$modelName"
    Write-Host "GAME_VISION_MODEL=$modelName"
}

Write-Host ""
Write-Host "完成。请彻底停止并重新启动 main.py。" -ForegroundColor Green
Write-Host "模型：$modelName / num_ctx=16384"
