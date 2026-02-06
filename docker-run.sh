#!/bin/bash

# OR-KAN Docker Execution Script (Arguments represent host paths)
# Usage:
#   ./docker-run.sh quality_control <host_input_dir> [--sequence TSE] [--ori coronal]
#   ./docker-run.sh quality_control_for_recon <host_input_dir> <host_output_dir>

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="or-kan:latest"
CONTAINER_NAME="or-kan-container"

# Check if Docker image exists
if ! docker images | grep -q "or-kan"; then
    echo "Building Docker image..."
    docker build -t $IMAGE_NAME $SCRIPT_DIR
fi

# Stop and remove existing container if present
if docker ps -a | grep -q $CONTAINER_NAME; then
    echo "Stopping and removing existing container..."
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
fi

# Parse arguments
SCRIPT=$1
shift || true

case "$SCRIPT" in
    quality_control)
        if [ -z "$1" ]; then
            echo "Usage: $0 quality_control <host_input_dir> [--sequence TSE] [--ori axial|coronal|sagittal]"
            echo "Example: $0 quality_control /data/my_fetal_scans"
            echo "         $0 quality_control /data/my_fetal_scans --sequence TSE --ori coronal"
            exit 1
        fi
        if [ ! -d "$1" ]; then
            echo "Error: Input directory does not exist: $1"
            exit 1
        fi
        HOST_INPUT="$(cd "$1" && pwd)"
        shift || true
        # Default output directory is inside the script directory if not specified
        mkdir -p "$SCRIPT_DIR/output"
        HOST_OUTPUT="$(cd "$SCRIPT_DIR/output" && pwd)"
        
        VOLUMES="-v $HOST_INPUT:/app/input:ro \
                 -v $SCRIPT_DIR/checkpoint:/app/checkpoint:ro \
                 -v $HOST_OUTPUT:/app/output"
        PYTHON_ARGS="--input_dir /app/input $*"
        ;;
        
    quality_control_for_recon)
        if [ -z "$1" ] || [ -z "$2" ]; then
            echo "Usage: $0 quality_control_for_recon <host_input_dir> <host_output_dir>"
            echo "Example: $0 quality_control_for_recon /data/subject_01 /data/out_subject_01"
            exit 1
        fi
        if [ ! -d "$1" ]; then
            echo "Error: Input directory does not exist: $1"
            exit 1
        fi
        HOST_INPUT="$(cd "$1" && pwd)"
        mkdir -p "$2"
        HOST_OUTPUT="$(cd "$2" && pwd)"
        
        VOLUMES="-v $HOST_INPUT:/app/input:ro \
                 -v $SCRIPT_DIR/checkpoint:/app/checkpoint:ro \
                 -v $HOST_OUTPUT:/app/output"
        PYTHON_ARGS="--input_dir /app/input --output_dir /app/output"
        ;;
        
    *)
        echo "Unknown script mode: $SCRIPT"
        echo "Supported modes: quality_control, quality_control_for_recon"
        exit 1
        ;;
esac

echo "Executing: python $SCRIPT.py $PYTHON_ARGS"
echo "Mounting: Input $HOST_INPUT -> /app/input, Output $HOST_OUTPUT -> /app/output"

docker run --rm \
    --name $CONTAINER_NAME \
    --gpus all \
    $VOLUMES \
    -e PYTHONUNBUFFERED=1 \
    $IMAGE_NAME \
    python $SCRIPT.py $PYTHON_ARGS