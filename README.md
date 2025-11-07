# RGB-SClusterFormer for Image Classification

This repository contains an implementation of the SClusterFormer architecture, adapted for standard RGB image classification. [cite_start]The original model was designed for Hyperspectral Image (HSI) analysis[cite: 7]. This version removes the HSI-specific components (e.g., 3D convolutions, spectral attention) and utilizes the spatial branch (SpaCA) for processing 2D images.

## Project Structure

```
.
├── model.py            # Contains all neural network modules
├── utils.py            # Data loading and utility functions
├── train.py            # Main script to run training and validation
├── requirements.txt    # Python package dependencies
└── README.md           # This file
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)<your-username>/<your-repo-name>.git
    cd <your-repo-name>
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Dataset

The model expects a dataset organized in the `ImageFolder` format:

```
<data_path>/
├── train/
│   ├── class_a/
│   │   ├── xxx.png
│   │   ├── xxy.png
│   │   └── ...
│   └── class_b/
│       ├── zzz.png
│       └── ...
└── test/
    ├── class_a/
    │   ├── 123.png
    │   └── ...
    └── class_b/
        ├── 456.png
        └── ...
```

## Training

Run the `train.py` script to start training. You can pass arguments to customize the run.

```bash
python train.py --data_path /path/to/your/dataset --epochs 50 --batch_size 32
```

### Arguments:
-   `--data_path`: (Required) Path to the root of your dataset.
-   `--img_size`: Input image resolution (default: 224).
-   `--batch_size`: Training batch size (default: 32).
-   `--epochs`: Number of epochs to train for (default: 50).
-   `--learning_rate`: Optimizer learning rate (default: 0.001).

The best performing model checkpoint will be saved as `best_model.pth`.