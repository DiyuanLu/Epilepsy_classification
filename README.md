# Epilepsy Classification Using Neural Networks

This repository contains the `nn_classification.py` script, which implements a neural network model for classifying epilepsy from EEG data. The model is designed to distinguish between epileptic and non-epileptic signals, aiding in the diagnosis and study of epilepsy.

## Features

- **Data Preprocessing**: Handles raw EEG data, including normalization and segmentation.
- **Model Architecture**: Utilizes a neural network tailored for time-series classification.
- **Training and Evaluation**: Includes functions for training the model and evaluating its performance on test datasets.

## Requirements

- Python 3.x
- TensorFlow or PyTorch (depending on the implementation)
- NumPy
- Pandas
- Scikit-learn

## Usage

1. **Clone the repository**:

   ```bash
   git clone https://github.com/DiyuanLu/Epilepsy_classification.git
   ```

2. **Navigate to the directory**:
  ```bash
  cd Epilepsy_classification
```

3. **Install the required packages**:
  ```bash
  pip install -r requirements.txt
  ```

4. **Run the script**:
  ```bash
  python nn_classification.py
```
