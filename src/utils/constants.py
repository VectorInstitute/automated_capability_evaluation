"""Module containing constants used throughout the codebase."""

import os


BASE_ARTIFACTS_DIR = "/fs01/projects/aieng/public/ace_artifacts"

SEED_CAPABILITIES_SCORE_DIR = os.path.join(
    BASE_ARTIFACTS_DIR, "seed_capabilities_results"
)
NON_SEED_CAPABILITIES_SCORE_DIR = os.path.join(
    BASE_ARTIFACTS_DIR, "capabilities_results"
)
