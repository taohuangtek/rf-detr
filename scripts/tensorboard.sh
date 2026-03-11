#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: ./start_tb.sh [path_to_logs]"
    echo "Example: ./start_tb.sh /workspace/output/tensorboard_logs"
    exit 1
fi

LOG_DIR=$1

echo "[*] Starting TensorBoard..."
echo "[*] Log directory: $LOG_DIR"
echo "[*] Port: 6006"
echo "[*] Binding to all interfaces..."

tensorboard --logdir="$LOG_DIR" --port=6006 --bind_all
