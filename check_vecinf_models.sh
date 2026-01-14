#!/bin/bash
# Helper script to check available Vector Inference models
# Run this on the cluster to see what model names are available

echo "=========================================="
echo "Checking Vector Inference availability..."
echo "=========================================="

if ! command -v vec-inf &> /dev/null; then
    echo "ERROR: vec-inf command not found!"
    echo "Make sure you're on the Vector cluster and vec-inf is in your PATH"
    exit 1
fi

echo ""
echo "vec-inf version:"
vec-inf --version 2>&1 || echo "Version check failed"

echo ""
echo "=========================================="
echo "Available models (JSON format):"
echo "=========================================="
vec-inf list --json-mode 2>/dev/null | python3 -m json.tool 2>/dev/null || vec-inf list

echo ""
echo "=========================================="
echo "Looking for Qwen models:"
echo "=========================================="
vec-inf list --json-mode 2>/dev/null | grep -i qwen || vec-inf list | grep -i qwen || echo "No Qwen models found in list"

echo ""
echo "=========================================="
echo "To run evaluation, use one of the model names above"
echo "Example: python3 -m src.task_solve_models.run_single_agent --batch-file evaluation_batch.json --model <MODEL_NAME>"
echo "=========================================="

