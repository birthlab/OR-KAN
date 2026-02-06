#!/bin/bash

# Docker container entrypoint script
# Supports direct execution of Python scripts or entering interactive shell

set -e

# Execute command directly if arguments are provided
if [ $# -gt 0 ]; then
    exec "$@"
else
    # Otherwise, enter interactive bash
    exec /bin/bash
fi