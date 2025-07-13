# Virtual Environment Setup Script for TFG Trading Bot Project (Windows)

Write-Host "Setting up virtual environment for TFG Trading Bot..." -ForegroundColor Green

# Remove existing virtual environment if it exists
if (Test-Path "venv") {
    Write-Host "Removing existing virtual environment..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force "venv"
}

# Create virtual environment with Python 3.9
Write-Host "Creating virtual environment with Python 3.9..." -ForegroundColor Green
python -m venv venv

# Check if virtual environment was created successfully
if (-not (Test-Path "venv")) {
    Write-Host "Failed to create virtual environment. Please make sure Python is installed and accessible." -ForegroundColor Red
    exit 1
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
& ".\venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Green
python -m pip install --upgrade pip

# Install requirements
Write-Host "Installing requirements from requirements.txt..." -ForegroundColor Green
pip install -r requirements.txt

Write-Host "Virtual environment setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "IMPORTANT: For web scraping functionality, you'll need Chrome or Chromium browser installed." -ForegroundColor Yellow
Write-Host "On Windows, you can install Chrome from:" -ForegroundColor Yellow
Write-Host "  https://www.google.com/chrome/"
Write-Host "Or install via Chocolatey:" -ForegroundColor Yellow
Write-Host "  choco install googlechrome"
Write-Host "Or install via winget:" -ForegroundColor Yellow
Write-Host "  winget install Google.Chrome"
Write-Host ""
Write-Host "To activate the environment in the future, run:" -ForegroundColor Cyan
Write-Host "  .\venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "To deactivate the environment, run:" -ForegroundColor Cyan
Write-Host "  deactivate"
Write-Host ""
Write-Host "Note: If you encounter execution policy issues, run:" -ForegroundColor Yellow
Write-Host "  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser"
