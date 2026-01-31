# Client Code Structure Analysis

This document provides a detailed overview of the `mini-ml` project structure, explaining the purpose of key directories and files.

## Directory Layout

```
ml-j/
├── config.yaml          # Main configuration file (Hyperparameters, Paths, Environment)
├── main.ipynb           # Interactive playground / Sandbox
├── pyproject.toml       # Project metadata and dependencies (uv/pep621)
├── docs/                # Documentation
├── outputs/             # Training artifacts (checkpoints, logs, visual results)
├── src/
│   └── mini_ml/         # Main package source
│       ├── core/        # Core execution logic
│       ├── models/      # Neural network architectures, datasets, and transforms
│       └── utils/       # Helper utilities (Metrics, Config parsing, Registration)
└── scripts/             # Helper scripts (e.g., TensorBoard launch)
```

## Key Modules

### 1. Core Logic (`src/mini_ml/core/`)

This directory contains the high-level orchestration code for training and inference.

- **`trainer.py`**:
    - **Purpose**: Encapsulates the entire training lifecycle.
    - **Responsibilities**: Environment setup, model/optimizer/loss instantiation (via factory), training loop implementation, validation loop, logging (TensorBoard/Console), and checkpoint saving.
    - **Key Class**: `Trainer`

- **`train.py`**:
    - **Purpose**: Entry point for training.
    - **Responsibilities**: CLI parsing, Config loading, delegating execution to `Trainer`.

- **`predict.py`**:
    - **Purpose**: Unified entry point for inference.
    - **Responsibilities**:
        - Supports **PyTorch** and **ONNX** backends.
        - Loads model configuration and weights.
        - Preprocesses input images (using reused validation transforms).
        - Post-processes model output (handling dictionary/list outputs).

- **`factory.py`**:
    - **Purpose**: Dynamic object creation.
    - **Responsibilities**: Instantiates Models, Datasets, Transforms, Optimizers, and Loss functions based on the `config.yaml` description.

### 2. Models & Data (`src/mini_ml/models/`)

- **`datasets/`**: Dataset implementations (e.g., `PetDataset` for Oxford-IIIT Pet).
- **`transforms/`**: Data augmentation and preprocessing pipelines.
    - **`compose.py`**: Sequential application of transforms (supports optional mask).
    - **`custom_transforms.py`**: Implementations (Resize, Normalize, Flip, ToTensor) adapting standard operations for segmentation tasks (handling image+mask pairs).

### 3. Utilities (`src/mini_ml/utils/`)

- **`metrics.py`**:
    - **Purpose**: Evaluation metrics.
    - **Key Class**: `MeanIoU` (Computes mIoU and Accuracy using confusion matrix).
- **`config.py`**: Pydantic models for strictly typed configuration validation.

## Configuration Flow

The project uses a configuration-driven approach:
1.  **`config.yaml`** defines the experiment.
2.  **`train.py`** loads this YAML into a Pydantic `AppConfig` object.
3.  **`Trainer`** uses the config to Request components from **`factory.py`**.
4.  **`factory.py`** uses the Registry pattern (implied) or direct mapping to instantiate Python classes.
