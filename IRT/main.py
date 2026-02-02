"""Main script for IRT analysis of ACE evaluation scores."""

import json
import logging
import os
from typing import Dict

import hydra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

from IRT.load_scores import (
    create_response_matrix,
    extract_question_responses,
    load_score_files,
)
from IRT.irt_analysis import calculate_statistics, fit_3pl_irt

matplotlib.use("Agg")  # Use non-interactive backend

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="cfg",
    config_name="irt_config"
)
def main(cfg: DictConfig) -> None:
    """Main function for IRT analysis."""
    print("="*80)
    print("STARTING IRT ANALYSIS")
    print("="*80)
    logger.info("Starting IRT analysis...")
    
    # Load score files
    scores_dir = cfg.data_cfg.scores_dir
    print(f"\n[STEP 1] Loading score files from: {scores_dir}")
    logger.info(f"Loading scores from: {scores_dir}")
    
    score_data = load_score_files(scores_dir)
    print(f"  -> Loaded {len(score_data)} score files")
    if not score_data:
        print("  -> ERROR: No score data loaded. Check the scores directory path.")
        logger.error("No score data loaded. Check the scores directory path.")
        return
    
    # Extract question responses
    print(f"\n[STEP 2] Extracting question responses from {len(score_data)} files...")
    response_data, question_info = extract_question_responses(score_data)
    print(f"  -> Extracted {len(question_info)} unique questions")
    print(f"  -> Extracted {len(response_data)} model-question responses")
    if not response_data:
        print("  -> ERROR: No response data extracted.")
        logger.error("No response data extracted.")
        return
    
    # Create response matrix
    print(f"\n[STEP 3] Creating response matrix...")
    response_matrix, model_names, question_ids = create_response_matrix(response_data)
    print(f"  -> Matrix shape: {len(question_ids)} questions x {len(model_names)} models")
    print(f"  -> Models: {', '.join(model_names)}")
    
    # Calculate basic statistics
    print(f"\n[STEP 4] Calculating basic statistics...")
    logger.info("Calculating statistics...")
    stats = calculate_statistics(response_matrix, question_ids, model_names)
    print(f"  -> Overall accuracy: {stats['overall']['accuracy']:.3f}")
    print(f"  -> Total responses: {stats['overall']['total_responses']}")
    print(f"  -> Correct responses: {stats['overall']['correct_responses']}")
    
    # Fit IRT model
    print(f"\n[STEP 5] Fitting IRT model (this may take a while)...")
    logger.info("Fitting IRT model...")
    try:
        irt_results = fit_3pl_irt(
            response_matrix,
            question_ids,
            model_names,
            max_iterations=cfg.irt_cfg.max_iterations,
            quadrature_n=cfg.irt_cfg.quadrature_n,
            model_type=cfg.irt_cfg.model_type,
        )
        print(f"  -> IRT model fitting completed successfully")
        print(f"  -> Model type: {irt_results.get('model_info', {}).get('model_type', 'Unknown')}")
    except Exception as e:
        print(f"  -> ERROR: Failed to fit IRT model: {e}")
        logger.error(f"Error fitting IRT model: {e}")
        logger.error("Make sure girth is installed: pip install girth")
        import traceback
        traceback.print_exc()
        return
    
    # Combine results
    results = {
        'irt_parameters': irt_results,
        'statistics': stats,
        'question_info': question_info,
        'model_names': model_names,
        'question_ids': question_ids,
    }
    
    # Save results
    print(f"\n[STEP 6] Saving results...")
    output_dir = cfg.output_cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"  -> Output directory: {output_dir}")
    
    output_file = os.path.join(output_dir, cfg.output_cfg.output_filename)
    print(f"  -> Writing to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  -> Results saved successfully!")
    logger.info(f"Results saved to: {output_file}")
    
    # Create and save plots for item parameters
    print(f"\n[STEP 7] Creating plots for item parameter distributions...")
    try:
        plot_file = os.path.join(output_dir, "irt_item_parameters_distributions.png")
        create_item_parameter_plots(irt_results, plot_file)
        print(f"  -> Plots saved to: {plot_file}")
        logger.info(f"Plots saved to: {plot_file}")
    except Exception as e:
        print(f"  -> Warning: Failed to create plots: {e}")
        logger.warning(f"Failed to create plots: {e}")
        import traceback
        traceback.print_exc()
    
    # Print summary
    print("\n" + "="*80)
    print("IRT ANALYSIS SUMMARY")
    print("="*80)
    print(f"Total questions: {len(question_ids)}")
    print(f"Total models: {len(model_names)}")
    print(f"Overall accuracy: {stats['overall']['accuracy']:.3f}")
    print(f"Total responses: {stats['overall']['total_responses']}")
    print(f"Correct responses: {stats['overall']['correct_responses']}")
    print(f"\nModel names: {', '.join(model_names)}")
    
    item_params = irt_results.get('item_parameters', {})
    
    print("\n" + "="*80)
    print(f"Full results saved to: {output_file}")
    plot_file = os.path.join(output_dir, "irt_item_parameters_distributions.png")
    print(f"Item parameter plots saved to: {plot_file}")
    print("="*80)


def create_item_parameter_plots(irt_results: Dict, output_path: str) -> None:
    """Create histogram plots for item parameter distributions.
    
    Args:
        irt_results: Dictionary containing IRT results with item_parameters
        output_path: Path to save the plot file
    """
    item_params = irt_results.get('item_parameters', {})
    model_info = irt_results.get('model_info', {})
    model_type = model_info.get('model_type', 'Unknown model')
    
    if not item_params:
        logger.warning("No item parameters found for plotting")
        return
    
    difficulties = []
    discriminations = []
    guessings = []
    
    for question_id, params in item_params.items():
        diff = params.get("difficulty")
        disc = params.get("discrimination")
        guess = params.get("guessing")
        
        if diff is not None and not np.isnan(diff):
            difficulties.append(diff)
        if disc is not None and not np.isnan(disc):
            discriminations.append(disc)
        if guess is not None and not np.isnan(guess):
            guessings.append(guess)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"IRT Item Parameter Distributions ({model_type})",
        fontsize=16,
        fontweight="bold",
    )
    
    ax1 = axes[0]
    if difficulties:
        ax1.hist(difficulties, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        ax1.set_xlabel('Difficulty (b)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title(f'Difficulty Distribution (n={len(difficulties)})', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.axvline(np.mean(difficulties), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(difficulties):.3f}')
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Difficulty Distribution', fontsize=12)
    
    ax2 = axes[1]
    if discriminations:
        ax2.hist(discriminations, bins=50, edgecolor='black', alpha=0.7, color='lightgreen')
        ax2.set_xlabel('Discrimination (a)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title(f'Discrimination Distribution (n={len(discriminations)})', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.axvline(np.mean(discriminations), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(discriminations):.3f}')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Discrimination Distribution', fontsize=12)
    
    ax3 = axes[2]
    if guessings:
        ax3.hist(guessings, bins=50, edgecolor='black', alpha=0.7, color='salmon')
        ax3.set_xlabel('Guessing (c)', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.set_title(f'Guessing Parameter Distribution (n={len(guessings)})', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.axvline(np.mean(guessings), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(guessings):.3f}')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Guessing Parameter Distribution', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Item parameter plots saved to {output_path}")


if __name__ == "__main__":
    main()

