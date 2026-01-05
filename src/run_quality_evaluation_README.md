# Quality Evaluation Script

`run_quality_evaluation.py` computes benchmark-level quality metrics from existing evaluation scores.

## Overview

This script analyzes model performance scores to compute several quality metrics:

- **Difficulty**: Measures how hard the benchmark is (`1 - max(accuracy)`)
- **Separability**: Measures how well the benchmark distinguishes between models (mean absolute deviation of accuracies)
- **Consistency**: Measures stability of model performance across different dataset generations (`1 - mean(std(performance across generations))`)
- **Novelty**: Measures how much new information the dataset reveals compared to prior benchmarks (`1 - rank_correlation(predicted, actual)`)

## Usage

```bash
python src/run_quality_evaluation.py
```

The script uses Hydra for configuration management. Configuration is specified in `src/cfg/run_quality_evaluation_cfg.yaml`.

## Configuration

Edit `src/cfg/run_quality_evaluation_cfg.yaml`:

```yaml
quality_eval_cfg:
  # Absolute path to directory containing per-model score folders
  scores_root_dir: "/path/to/scores"
  
  # Fallback: if scores_root_dir not set, uses:
  # {BASE_ARTIFACTS_DIR}/{scores_subdir}/{run_id}
  scores_subdir: "scores"
  
  # Optional: List of prior datasets for novelty computation
  prior_datasets:
    - "/path/to/prior_dataset1"
    - "/path/to/prior_dataset2"
```

## Data Structure

The script expects a root directory containing **per-model subdirectories**. Two structures are supported:

### Structure 1: With Multiple Generations (for Consistency)

```
scores_root_dir/
├── model1/
│   ├── generation1/          # First dataset generation
│   │   └── .../*.json files (recursively)
│   ├── generation2/          # Second dataset generation
│   │   └── .../*.json files
│   └── generation3/
│       └── .../*.json files
├── model2/
│   └── ... (same structure)
```

**Behavior:**
- Computes average accuracy **per generation** for each model
- **Consistency** is computed from generation-to-generation variation
- **Difficulty** and **Separability** use the **average across all generations**

### Structure 2: Without Generations (Single Dataset)

```
scores_root_dir/
├── model1/
│   └── .../*.json files (recursively, any nesting allowed)
├── model2/
│   └── .../*.json files
```

**Behavior:**
- Walks all JSON files recursively under each model directory
- Computes average accuracy per model
- **Consistency** is NOT computed (no generations available)
- **Difficulty** and **Separability** are computed from average accuracies

## JSON File Format

Each `.json` file must follow the Inspect AI evaluation format:

```json
{
  "results": {
    "scores": [
      {
        "metrics": {
          "accuracy": {
            "value": 0.75
          }
        }
      }
    ]
  }
}
```

## Metrics

### Difficulty

Measures how difficult the benchmark is for models:

```
difficulty = 1 - max(accuracy across all models)
```

- Range: [0, 1]
- Higher values = harder benchmark

### Separability

Measures how well the benchmark distinguishes between models:

```
separability = mean(|accuracy_i - mean(accuracies)|)
```

- Range: [0, 1]
- Higher values = better model discrimination

### Consistency

Measures stability of model performance across dataset generations:

```
consistency = 1 - (1/n) * Σ std(performance(m_i) across generations)
```

- Range: [0, 1]
- Higher values = more stable/consistent performance
- **Only computed** when multiple generations are detected

### Novelty

Measures how much new information the dataset reveals compared to prior benchmarks:

```
1. Predict current accuracies from prior datasets using linear regression
2. Compute rank correlation between predicted and actual rankings
3. novelty = 1 - rank_correlation
```

- Range: [0, 1]
- Higher values = more novel/unpredictable performance patterns
- **Only computed** when `prior_datasets` are specified in config

## Prior Datasets (for Novelty)

Prior datasets should have the **same structure** as the main dataset.

**Important:** Prior dataset directories should be **separate** from the main `scores_root_dir` to avoid being treated as models.

Example:
```
data/
├── scores_sample/          # Main dataset
│   ├── model1/
│   └── model2/
└── scores_sample/
    └── math-500/          # Prior dataset (separate directory)
        ├── model1/
        └── model2/
```

**Requirements:**
- All prior datasets must have the same set of models as the current dataset
- Models must be consistent across all datasets for novelty computation

## Output

The script logs all computed metrics:

```
[INFO] Model 'model1' mean accuracy over 3 generations: 0.7500
[INFO] Model 'model2' mean accuracy over 3 generations: 0.6500
[INFO] Benchmark difficulty: 0.2500
[INFO] Benchmark separability: 0.0500
[INFO] Benchmark consistency: 0.9200
[INFO] Benchmark novelty: 0.5000
```

