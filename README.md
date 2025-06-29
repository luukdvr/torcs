# TORCS AI Racing Bot

A PyTorch-based AI model for autonomous racing in TORCS (The Open Racing Car Simulator).

## Installation

Install dependencies from requirements file:

```bash
pip install -r requiments.txt
```

## Usage

### 1. Train the Model

Train the neural network on collected racing data:

```bash
python train_model.py
```

This will:
- Load training data from `combined_data_cleaned.csv`
- Train a neural network model
- Save the trained model as `expert_model.pt`
- Save the feature scaler as `expert_scaler.save`

### 2. Run Predictions

Start the TORCS server first, then run the prediction model:

```bash
python predict_model.py
```

This will:
- Connect to TORCS server (localhost:3001)
- Load the trained model and scaler
- Make real-time driving predictions
- Control the car autonomously

## Requirements

- TORCS simulator running on localhost:3001
- Training data in `combined_data_cleaned.csv` format
- Python 3.7+ with PyTorch, scikit-learn, pandas, numpy