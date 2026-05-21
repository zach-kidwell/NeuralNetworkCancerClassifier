# Cancer Image Classification using CNNs

A complete PyTorch project for classifying histopathology image tiles as cancerous or non-cancerous using a convolutional neural network. The project is structured for a GitHub portfolio and includes training, validation, testing, model checkpointing, visualizations, evaluation metrics, and a Streamlit web app.

## Project Overview

This project trains a CNN on the Kaggle **Histopathologic Cancer Detection** dataset. Each image is a 96x96 histopathology tile, and the task is binary classification:

- `0`: No Cancer Detected
- `1`: Cancer Detected

The model learns visual tissue patterns from image tiles and outputs a class prediction with a confidence score.

## Dataset

Dataset: [Kaggle Histopathologic Cancer Detection](https://www.kaggle.com/competitions/histopathologic-cancer-detection)

Expected local layout:

```text
data/
  raw/
    train/
      <image_id>.tif
    test/
      <image_id>.tif
    train_labels.csv
```

Download with the Kaggle CLI:

```bash
pip install kaggle
kaggle competitions download -c histopathologic-cancer-detection -p data/raw
```

Or use the included helper:

```bash
python -m src.download_data
```

You need a Kaggle API token configured at `~/.kaggle/kaggle.json`.

## Technologies Used

- Python
- PyTorch and TorchVision
- scikit-learn
- pandas and NumPy
- Matplotlib and Seaborn
- Streamlit
- Kaggle CLI

## Project Structure

```text
Cancer-Image-Classification-using-CNNs/
  app.py
  data/
  models/
  notebooks/
  outputs/
  src/
    checkpoint.py
    config.py
    dataset.py
    download_data.py
    evaluate.py
    model.py
    predict.py
    train.py
    utils.py
    visualize.py
  README.md
  requirements.txt
```

## Model Architecture

The default model is a compact CNN designed for 96x96 RGB tissue tiles:

- Four convolutional blocks
- Each block uses convolution, batch normalization, ReLU activation, and max pooling
- Dropout in the classifier head to reduce overfitting
- Final linear layer outputs two class logits

This architecture is intentionally beginner-friendly while still using professional ML building blocks commonly found in production research code.

## Data Pipeline

The training pipeline includes:

- CSV label loading
- Stratified train/validation/test split
- Image preprocessing
- Data augmentation with flips, rotations, and color jitter
- Normalization
- PyTorch `Dataset` and `DataLoader` abstractions
- CUDA acceleration when available

## Training

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the CNN:

```bash
python -m src.train --epochs 10 --batch-size 64 --learning-rate 0.001
```

The best validation checkpoint is saved to:

```text
models/cancer_cnn.pth
```

Training curves are saved to:

```text
outputs/figures/loss_curve.png
outputs/figures/accuracy_curve.png
```

## Evaluation

Run evaluation on the held-out test split:

```bash
python -m src.evaluate
```

Evaluation outputs include:

- Accuracy
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix
- Sample prediction visualization
- Incorrect prediction visualization

Saved outputs:

```text
outputs/metrics.json
outputs/figures/confusion_matrix.png
outputs/predictions/sample_predictions.png
outputs/predictions/incorrect_predictions.png
```

## Streamlit Web App

Launch the app:

```bash
streamlit run app.py
```

The app lets a user upload a histopathology image and returns:

- `Cancer Detected` or `No Cancer Detected`
- Confidence percentage
- Cancer and non-cancer probability bars

Train the model first so `models/cancer_cnn.pth` exists.

## Hosting on Render

Do not commit the full Kaggle dataset to GitHub. It is large, slows deployment,
and should be downloaded through Kaggle credentials. For a hosted portfolio demo,
train locally and commit only:

- `models/cancer_cnn.pth`
- a small demo subset in `data/demo`

Create the hostable demo subset after downloading the Kaggle data and training:

```bash
py -m src.download_data
py -m src.train --epochs 10
py -m src.export_demo_assets --sample-count 50
```

If you only need the website UI to run before training the real Kaggle model,
create a tiny synthetic demo dataset:

```bash
py -m src.create_synthetic_demo --sample-count 24
```

When `models/cancer_cnn.pth` is missing, the app clearly labels predictions as
demo fallback results. Train the CNN and include the checkpoint for real model
predictions.

The app automatically uses:

1. `data/raw/train_labels.csv` and `data/raw/train` when the full local dataset exists.
2. `data/demo/labels.csv` and `data/demo/images` when running as a hosted demo.

Recommended Render settings:

```text
Build Command: pip install -r requirements.txt
Start Command: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

## Results

After training, update this section with your actual test metrics from `outputs/metrics.json`.

Example format:

| Metric | Score |
| --- | ---: |
| Accuracy | TBD |
| Precision | TBD |
| Recall | TBD |
| F1-score | TBD |

## Notes for Portfolio Use

This repository demonstrates:

- End-to-end ML project structure
- CNN image classification with PyTorch
- Reproducible training and evaluation scripts
- Model checkpoint saving and loading
- Clear visual outputs for model performance
- A deployable Streamlit inference interface

## Disclaimer

This project is for educational and portfolio purposes only. It is not a clinical diagnostic tool.
