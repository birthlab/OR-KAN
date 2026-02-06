# OR-KAN Docker User Guide

This document describes how to run the OR-KAN project using Docker.

## Prerequisites

- Docker (version 20.10+)
- NVIDIA Container Toolkit (install [nvidia-docker2](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU usage)

## Preparation Before Use

1. **Ensure the `checkpoint` directory exists** and contains the pre-trained model `OR_KAN_weight.pth`.
2. **Prepare input data** (default location: `data_example_with_mask` directory).
3. **Create the output directory**: `mkdir -p output`.

## Quick Start

### Method 1: Using Makefile (Recommended)

```bash
make build          # Build the image
make run-qc         # Run quality assessment
make run-recon      # Run reconstruction pipeline with quality filtering
make shell          # Enter interactive container
```

`INPUT_DIR` and `OUTPUT_DIR` are paths on the host machine, with defaults set to `./data_example_with_mask`and `./output`, respectively. Examples of customizing host machine data paths or passing parameters:

```bash
make run-qc INPUT_DIR=/path/to/your/data
make run-recon INPUT_DIR=/path/to/input OUTPUT_DIR=/path/to/output
make run-qc-custom INPUT_DIR=/path/to/data ARGS='--sequence TSE --ori coronal'
```

For more commands, see `make help`.

### Method 2: Using the Convenience Script docker-run.sh

```bash
# Grant execution permission for first-time use
chmod +x docker-run.sh   

# Quality assessment
./docker-run.sh quality_control data_example_with_mask

# Optional: Specify sequence and orientation
./docker-run.sh quality_control data_example_with_mask --sequence TSE --ori coronal

# Reconstruction pipeline with quality filtering
./docker-run.sh quality_control_for_recon data_example_with_mask output
```

### Method 3: Using Docker Commands Directly

**Build the Image:**

```bash
docker build -t or-kan:latest .
```

**Run Quality Assessment:**

```bash
docker run --rm --gpus all \
    -v $(pwd)/data_example_with_mask:/app/input:ro \
    -v $(pwd)/checkpoint:/app/checkpoint:ro \
    -v $(pwd)/output:/app/output \
    or-kan:latest \
    python quality_control.py --input_dir /app/input
```

**Run Reconstruction Quality Screening:**

```bash
docker run --rm --gpus all \
    -v $(pwd)/data_example_with_mask:/app/input:ro \
    -v $(pwd)/checkpoint:/app/checkpoint:ro \
    -v $(pwd)/output:/app/output \
    or-kan:latest \
    python quality_control_for_recon.py --input_dir /app/input --output_dir /app/output
```

The `-v` flag follows the format `host_path:container_path`. Since Python runs inside the container, `--input_dir`/`--output_dir` must refer to container paths.

## Batch Processing Multiple Directories

```bash
for dir in /path/to/data/*/; do
    docker run --rm --gpus all \
        -v "$dir":/app/input:ro \
        -v $(pwd)/checkpoint:/app/checkpoint:ro \
        -v $(pwd)/output:/app/output \
        or-kan:latest \
        python quality_control.py --input_dir /app/input
done
```
