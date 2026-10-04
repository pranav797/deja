# Starts Déjà (with near-match logging) and Open WebUI connected to it, both on this computer only.
# Open http://127.0.0.1:8080 once it says "Application startup complete". Ctrl+C stops both.
# Live dashboard of what Déjà decides: http://127.0.0.1:8000/dashboard
# Open WebUI keeps its account, chats and settings in $HOME\.open-webui; Déjà's cache and log go in deja.db here.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$key = (Get-Content .env | Where-Object { $_ -match '^OPENAI_API_KEY=' }) -replace '^OPENAI_API_KEY=', ''
if (-not $key) { throw "Put OPENAI_API_KEY=... in .env first" }
$data = Join-Path $HOME ".open-webui"
New-Item -ItemType Directory -Force $data | Out-Null

$env:DEJA_LOG_CANDIDATES = "1"
$deja = Start-Process uv -ArgumentList "run --env-file .env --extra proxy deja-proxy" -NoNewWindow -PassThru

# Open WebUI reads these on first start, then keeps them in its own settings (change them in Admin Settings after).
$env:DATA_DIR = $data
$env:OPENAI_API_BASE_URL = "http://127.0.0.1:8000/v1"
$env:OPENAI_API_KEY = $key
$env:ENABLE_OLLAMA_API = "false"
# Background title/tag/follow-up calls share long templates, so Déjà would see different chats as near-duplicates.
$env:ENABLE_TITLE_GENERATION = "false"
$env:ENABLE_TAGS_GENERATION = "false"
$env:ENABLE_FOLLOW_UP_GENERATION = "false"
$env:ENABLE_AUTOCOMPLETE_GENERATION = "false"
$webui = Start-Process open-webui -ArgumentList "serve --host 127.0.0.1 --port 8080" -WorkingDirectory $data -NoNewWindow -PassThru

try {
    Wait-Process -Id $webui.Id
} finally {
    foreach ($p in $webui, $deja) { if (-not $p.HasExited) { taskkill /T /F /PID $p.Id | Out-Null } }
}
