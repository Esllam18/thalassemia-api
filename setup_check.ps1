# =============================================================
# Run this entire script in PowerShell from D:\Thalassemis_ai
# It will create your .env file and verify the setup.
# =============================================================

# Step 1: Create the .env file with correct content
$envContent = @"
MODEL_PATH=thalassemia_expert_model.pkl
ENCODER_PATH=label_encoder.pkl
HOST=0.0.0.0
PORT=8000
RELOAD=false
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8088
MAX_UPLOAD_BYTES=10485760
MCV_NORMAL_THRESHOLD=81.0
"@

$envContent | Out-File -FilePath ".env" -Encoding utf8 -NoNewline
Write-Host "✓ .env file created" -ForegroundColor Green

# Step 2: Verify .env was created
if (Test-Path ".env") {
    Write-Host "✓ .env exists" -ForegroundColor Green
    Write-Host "Contents:" -ForegroundColor Yellow
    Get-Content ".env"
} else {
    Write-Host "✗ .env was NOT created — check permissions" -ForegroundColor Red
}

# Step 3: Verify model files exist
if (Test-Path "thalassemia_expert_model.pkl") {
    Write-Host "✓ thalassemia_expert_model.pkl found" -ForegroundColor Green
} else {
    Write-Host "✗ thalassemia_expert_model.pkl NOT found — move it to D:\Thalassemis_ai\" -ForegroundColor Red
}

if (Test-Path "label_encoder.pkl") {
    Write-Host "✓ label_encoder.pkl found" -ForegroundColor Green
} else {
    Write-Host "✗ label_encoder.pkl NOT found — move it to D:\Thalassemis_ai\" -ForegroundColor Red
}

# Step 4: Verify main.py has the fix
$mainContent = Get-Content "main.py" -Raw
if ($mainContent -match "response_model=None") {
    Write-Host "✓ main.py has the response_model=None fix" -ForegroundColor Green
} else {
    Write-Host "✗ main.py is still the OLD version — download and replace it!" -ForegroundColor Red
    Write-Host "  The fixed main.py is in your Claude chat downloads" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "If all 5 checks show ✓, run: python main.py" -ForegroundColor Cyan
