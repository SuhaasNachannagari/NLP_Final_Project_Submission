#!/bin/bash

# CS490 NLP Final Project - Environment Setup
# Toxicity Classification in Gaming Chat

echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete. Activate the environment with: source venv/bin/activate"