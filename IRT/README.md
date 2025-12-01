# IRT Analysis for ACE Evaluation Scores

This module performs Item Response Theory (IRT) analysis on ACE evaluation scores using the
[`girth`](https://pypi.org/project/girth/) library. It currently supports 1PL (Rasch), 2PL,
and 3PL logistic models for item parameter estimation.

## Installation

Install the required IRT library:

```bash
pip install girth
```

## Usage

Run the IRT analysis from the project root:

```bash
cd IRT
python main.py
```

## Configuration

Edit `cfg/irt_config.yaml` to configure:
- `data_cfg.scores_dir`: Path to the ACE scores directory containing JSON evaluation files
- `output_cfg.output_dir`: Directory where results and plots will be written
- `output_cfg.output_filename`: Name of the output JSON file
- `irt_cfg.model_type`: IRT model type, one of `1PL`, `2PL`, or `3PL`
- `irt_cfg.max_iterations`: Maximum number of MML iterations for GIRTH
- `irt_cfg.quadrature_n`: Number of quadrature points used by GIRTH

## Output

The analysis produces:
- **Results JSON** (at `output_cfg.output_dir/output_cfg.output_filename`) containing:
  - **IRT Parameters**: Item difficulty, discrimination, and guessing parameters
    (guessing is fixed to 0 for 1PL/2PL; discrimination is fixed to 1 for 1PL)
  - **Statistics**: Question-level and model-level descriptive statistics
  - **Question Info**: Metadata for each question (task, ID, input, target)
- **Item-parameter plots**: A PNG file `irt_item_parameters_distributions.png` in
  `output_cfg.output_dir`, showing histograms of difficulty, discrimination, and guessing
  for the chosen PL model.

Note: Person abilities are estimated internally (for some diagnostics) but are **not**
saved to disk or printed in the console output.

## IRT Parameters Explained

- **Difficulty (b)**: How difficult the question is (higher = more difficult).
- **Discrimination (a)**: How well the question distinguishes between high- and low-ability models.
- **Guessing (c)**: Lower asymptote (probability of a correct answer by guessing). This is 0
  in 1PL and 2PL models, and estimated in 3PL models.

