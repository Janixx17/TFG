#!/bin/bash

# Virtual Environment Setup Script for TFG Trading Bot Project

echo "Setting up virtual environment for TFG Trading Bot..."

# Remove existing virtual environment if it exists
if [ -d "venv" ]; then
    echo "Removing existing virtual environment..."
    rm -rf venv
fi

# Create virtual environment with Python 3.9
echo "Creating virtual environment with Python 3.9..."
/usr/bin/python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements from requirements.txt..."
pip install -r requirements.txt

echo "Virtual environment setup complete!"
echo ""
echo "IMPORTANT: For web scraping functionality, you'll need Chrome or Chromium browser installed."
echo "On macOS, you can install Chrome via:"
echo "  brew install --cask google-chrome"
echo "Or Chromium via:"
echo "  brew install --cask chromium"
echo ""
echo "To activate the environment in the future, run:"
echo "source venv/bin/activate"
echo ""
echo "To deactivate the environment, run:"
echo "deactivate"