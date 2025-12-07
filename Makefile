# Makefile for DCGAN Image Generation Project
# Author: RSK World
# Website: https://rskworld.in
# Email: help@rskworld.in
# Phone: +91 93305 39277

.PHONY: help install test train clean setup

help:
	@echo "DCGAN Image Generation - Makefile Commands"
	@echo "Author: RSK World"
	@echo "Website: https://rskworld.in"
	@echo ""
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make test      - Run setup tests"
	@echo "  make train     - Start training"
	@echo "  make setup     - Create directories"
	@echo "  make clean     - Clean generated files"

install:
	pip install -r requirements.txt

test:
	python test_setup.py

train:
	python main.py

setup:
	mkdir -p data/custom outputs checkpoints
	@echo "Directories created"

clean:
	rm -rf outputs/*.png
	rm -rf checkpoints/*.pth
	@echo "Cleaned output files"

