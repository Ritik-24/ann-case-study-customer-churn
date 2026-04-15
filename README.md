# Bank Customer Churn Prediction 🏦
### Artificial Neural Network (ANN) Case Study

This project implements a Deep Learning model to predict whether a bank customer will leave (churn) or stay, based on various demographic and financial factors. This was developed as a Case Study for my B.Tech CSE curriculum.

## 📊 Performance
- **Test Accuracy:** 86.1%
- **Data Split:** 80% Training, 20% Testing (8,000 / 2,000 records)
- **Optimizer:** Adam
- **Loss Function:** Binary Cross-Entropy

## 🛠️ Project Structure
- `Case study.py`: The main training script including data preprocessing, ANN architecture, and evaluation.
- `app.py`: A Streamlit web application providing a user-friendly frontend for real-time predictions.
- `Artificial_Neural_Network_Case_Study_data.csv`: The dataset containing 10,000 customer records.
- `churn_model.h5`: The trained and saved ANN model.
- `scaler.joblib`: The saved StandardScaler object used for consistent data normalization.

## 🚀 Model Architecture
1. **Input Layer:** Receives 12 features (after One-Hot and Label Encoding).
2. **Hidden Layer 1:** 6 Neurons, ReLU Activation.
3. **Hidden Layer 2:** 6 Neurons, ReLU Activation.
4. **Output Layer:** 1 Neuron, Sigmoid Activation (outputs probability 0-1).
