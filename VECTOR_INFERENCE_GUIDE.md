# Running Qwen 32B with Vector Inference

## Quick Start

1. **SSH to the Vector cluster:**
   ```bash
   ssh alimehr@v1.vectorinstitute.ai
   ```

2. **Check available models (optional but recommended):**
   ```bash
   cd /path/to/automated_capability_evaluation
   ./check_vecinf_models.sh
   ```
   This will show you the exact model names available in Vector Inference.

3. **Run the evaluation:**

   **Option A: Submit as Slurm job (recommended for long runs):**
   ```bash
   sbatch run_qwen32b_vecinf.sh
   ```
   Check status: `squeue -u $USER`
   View output: `tail -f qwen32b_eval_<JOB_ID>.out`

   **Option B: Run directly (for testing):**
   ```bash
   python3 -m src.task_solve_models.run_single_agent \
     --batch-file evaluation_batch.json \
     --model Qwen2.5-32B-Instruct
   ```

## How It Works

The code automatically detects that `Qwen2.5-32B-Instruct` is a 32B model and uses Vector Inference:

1. Calls `vec-inf list` to verify the model is available
2. Launches the model via `vec-inf launch Qwen2.5-32B-Instruct`
3. Waits for the model to be ready (checks status every 5 seconds)
4. Gets the base URL and runs the evaluation
5. The model stays running until the evaluation completes

## Model Name Format

The model name must match exactly what `vec-inf list` returns. Common formats:
- `Qwen2.5-32B-Instruct`
- `qwen2.5-32b-instruct`
- `Qwen/Qwen2.5-32B-Instruct`

**Important:** Run `check_vecinf_models.sh` first to see the exact format!

## Troubleshooting

- **"Model not found"**: Check available models with `vec-inf list --json-mode`
- **"vec-inf not found"**: Make sure you're on the cluster and `vec-inf` is in your PATH
- **Connection errors**: The model might still be launching. Wait a few minutes and check status with `vec-inf status <JOB_ID>`

## Results

Results will be saved to:
```
src/task_solve_models/Results/results_Qwen2.5-32B-Instruct_evaluation_batch.json
```

