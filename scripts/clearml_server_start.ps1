# PowerShell script to start ClearML Server locally using Docker Compose

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

Set-Location $ProjectRoot

Write-Host "Starting ClearML Server..." -ForegroundColor Green
Write-Host ""

# Check if Docker is running
try {
    docker info | Out-Null
} catch {
    Write-Host "Error: Docker is not running. Please start Docker and try again." -ForegroundColor Red
    exit 1
}

# Check if docker-compose is available
$UseDockerCompose = $false
if (Get-Command docker-compose -ErrorAction SilentlyContinue) {
    $UseDockerCompose = $true
} elseif (docker compose version 2>$null) {
    $UseDockerCompose = $false
} else {
    Write-Host "Error: docker-compose or 'docker compose' not found. Please install Docker Compose." -ForegroundColor Red
    exit 1
}

# Start the server
if ($UseDockerCompose) {
    docker-compose -f docker-compose.clearml.yml up -d
} else {
    docker compose -f docker-compose.clearml.yml up -d
}

Write-Host ""
Write-Host "ClearML Server is starting..." -ForegroundColor Yellow
Write-Host ""
Write-Host "Web UI will be available at: http://localhost:8080" -ForegroundColor Cyan
Write-Host "API Server will be available at: http://localhost:8008" -ForegroundColor Cyan
Write-Host ""
Write-Host "To view logs, run:" -ForegroundColor Gray
if ($UseDockerCompose) {
    Write-Host '  docker-compose -f docker-compose.clearml.yml logs -f' -ForegroundColor Gray
} else {
    Write-Host '  docker compose -f docker-compose.clearml.yml logs -f' -ForegroundColor Gray
}
Write-Host ""
Write-Host "To stop the server, run:" -ForegroundColor Gray
if ($UseDockerCompose) {
    Write-Host '  docker-compose -f docker-compose.clearml.yml down' -ForegroundColor Gray
} else {
    Write-Host '  docker compose -f docker-compose.clearml.yml down' -ForegroundColor Gray
}
Write-Host ""
Write-Host "Waiting for server to be ready..." -ForegroundColor Yellow

Start-Sleep -Seconds 5

# Wait for server to be ready
$MaxWait = 120
$WaitTime = 0
while ($WaitTime -lt $MaxWait) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8080" -UseBasicParsing -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            Write-Host ""
            Write-Host "ClearML Server is ready!" -ForegroundColor Green
            Write-Host ""
            Write-Host "Next steps:" -ForegroundColor Cyan
            Write-Host "1. Open http://localhost:8080 in your browser" -ForegroundColor White
            Write-Host "2. Create an account (first user becomes admin)" -ForegroundColor White
            Write-Host "3. Go to Settings -> Workspace -> Create new credentials" -ForegroundColor White
            Write-Host "4. Run: clearml-init" -ForegroundColor White
            Write-Host "   Or set environment variables:" -ForegroundColor White
            Write-Host '   $env:CLEARML_API_HOST="http://localhost:8008"' -ForegroundColor Gray
            Write-Host '   $env:CLEARML_API_ACCESS_KEY="your-access-key"' -ForegroundColor Gray
            Write-Host '   $env:CLEARML_API_SECRET_KEY="your-secret-key"' -ForegroundColor Gray
            exit 0
        }
    } catch {
        # Server not ready yet
    }
    Start-Sleep -Seconds 2
    $WaitTime += 2
    Write-Host "." -NoNewline -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Warning: Server may still be starting. Check logs with:" -ForegroundColor Yellow
if ($UseDockerCompose) {
    Write-Host '  docker-compose -f docker-compose.clearml.yml logs' -ForegroundColor Gray
} else {
    Write-Host '  docker compose -f docker-compose.clearml.yml logs' -ForegroundColor Gray
}
exit 0
