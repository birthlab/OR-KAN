.PHONY: build run-qc run-recon shell clean help

# Docker Image Configuration
IMAGE_NAME = or-kan:latest
CONTAINER_NAME = or-kan-container

# Default Data Directories
INPUT_DIR = ./data_example_with_mask
OUTPUT_DIR = ./output
CHECKPOINT_DIR = ./checkpoint

help:
	@echo "OR-KAN Docker Usage:"
	@echo ""
	@echo "  make build          - Build Docker image"
	@echo "  make run-qc         - Run quality control (auto-detection)"
	@echo "  make run-qc-custom  - Run quality control (custom arguments)"
	@echo "  make run-recon      - Run reconstruction quality screening"
	@echo "  make shell          - Enter interactive container shell"
	@echo "  make clean          - Clean Docker resources"
	@echo ""
	@echo "Examples:"
	@echo "  make run-qc INPUT_DIR=/path/to/data"
	@echo "  make run-qc-custom ARGS='--sequence TSE --ori coronal'"

build:
	@echo "Building Docker image..."
	docker build -t $(IMAGE_NAME) .

run-qc:
	@echo "Running quality control..."
	docker run --rm --gpus all \
		-v $(abspath $(INPUT_DIR)):/app/input:ro \
		-v $(abspath $(CHECKPOINT_DIR)):/app/checkpoint:ro \
		-v $(abspath $(OUTPUT_DIR)):/app/output \
		$(IMAGE_NAME) \
		python quality_control.py --input_dir /app/input $(ARGS)

run-qc-custom:
	@echo "Running quality control (custom arguments)..."
	@if [ -z "$(ARGS)" ]; then \
		echo "Error: Please provide ARGS parameter, e.g.: make run-qc-custom ARGS='--sequence TSE --ori coronal'"; \
		exit 1; \
	fi
	docker run --rm --gpus all \
		-v $(abspath $(INPUT_DIR)):/app/input:ro \
		-v $(abspath $(CHECKPOINT_DIR)):/app/checkpoint:ro \
		-v $(abspath $(OUTPUT_DIR)):/app/output \
		$(IMAGE_NAME) \
		python quality_control.py --input_dir /app/input $(ARGS)

run-recon:
	@echo "Running reconstruction quality screening..."
	docker run --rm --gpus all \
		-v $(abspath $(INPUT_DIR)):/app/input:ro \
		-v $(abspath $(CHECKPOINT_DIR)):/app/checkpoint:ro \
		-v $(abspath $(OUTPUT_DIR)):/app/output \
		$(IMAGE_NAME) \
		python quality_control_for_recon.py --input_dir /app/input --output_dir /app/output

shell:
	@echo "Entering interactive container..."
	docker run --rm -it --gpus all \
		-v $(abspath $(INPUT_DIR)):/app/input:ro \
		-v $(abspath $(CHECKPOINT_DIR)):/app/checkpoint:ro \
		-v $(abspath $(OUTPUT_DIR)):/app/output \
		$(IMAGE_NAME) \
		/bin/bash

clean:
	@echo "Cleaning Docker resources..."
	-docker stop $(CONTAINER_NAME) 2>/dev/null || true
	-docker rm $(CONTAINER_NAME) 2>/dev/null || true
	-docker rmi $(IMAGE_NAME) 2>/dev/null || true
	@echo "Cleanup complete."