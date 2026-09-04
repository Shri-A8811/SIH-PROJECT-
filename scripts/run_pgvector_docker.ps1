# PowerShell script to spin up an air-gapped PostgreSQL + pgvector Docker container locally for Sovereign Workbench
Write-Host "🛡️ Starting Sovereign Air-Gapped PostgreSQL + pgvector Container..." -ForegroundColor Cyan

$CONTAINER_NAME = "sovereign-pgvector"
$PORT = "5432"
$DB_NAME = "sovereign_workbench"
$PASSWORD = $env:WORKBENCH_POSTGRES_PASSWORD
if ([string]::IsNullOrWhiteSpace($PASSWORD)) {
    throw "Set WORKBENCH_POSTGRES_PASSWORD before starting the local database."
}

# Check if container already exists
$existing = docker ps -a -q -f name=$CONTAINER_NAME

if ($existing) {
    Write-Host "Container '$CONTAINER_NAME' exists. Starting..." -ForegroundColor Yellow
    docker start $CONTAINER_NAME
} else {
    if (-not (docker image inspect pgvector/pgvector:pg16 2>$null)) {
        throw "Required image pgvector/pgvector:pg16 is not loaded. Import the approved offline OCI archive before launch."
    }
    Write-Host "Running preloaded 'pgvector/pgvector:pg16' on port $PORT..." -ForegroundColor Green
    docker run -d `
        --name $CONTAINER_NAME `
        -e POSTGRES_PASSWORD=$PASSWORD `
        -e POSTGRES_DB=$DB_NAME `
        -p ${PORT}:5432 `
        --pull never `
        pgvector/pgvector:pg16
}

Write-Host "✅ PostgreSQL + pgvector is running at localhost:$PORT with database '$DB_NAME'" -ForegroundColor Green
Write-Host "Next step: Run 'python scripts/init_postgres_pgvector.py' to verify tables." -ForegroundColor Cyan
