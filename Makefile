# Variables
VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(PYTHON) -m pip
SCRIPT := main.py

# Targets
.PHONY: all setup run clean help

# Default target: sets up the environment, installs dependencies, and runs the script
all: setup run

# Help target: displays available commands
help:
	@echo "Available commands:"
	@echo "  make setup         - Set up virtual environment and install dependencies"
	@echo "  make run           - Run the Python script with the specified folder path"
	@echo "  make clean         - Remove virtual environment and cache files"
	@echo "  make help          - Display this help message"

# Set up virtual environment and install dependencies
setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# Source venv linux:
# source .venv/bin/activate

# Source venv windows:
# .venv\Scripts\activate

# Run the Python script with the specified folder path
run:
	$(PYTHON) $(SCRIPT)

# Clean up virtual environment and cache files
clean:
	rm -rf $(VENV) __pycache__ *.pyc *.pyo
