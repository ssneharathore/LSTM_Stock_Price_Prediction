# 📈 LSTM Stock Price Prediction using Technical Indicators

Predict the future... well, at least stock prices!  
This deep learning project leverages **LSTM (Long Short-Term Memory)** models combined with **technical analysis indicators** like **MACD** and **RSI** to forecast stock price trends. Built with Python, visualized interactively, and deployed via Streamlit.

---

## 🚀 Demo

👉 [Run the Web App on Streamlit](https://your-username-your-app-name.streamlit.app)

---

## 🧠 Project Overview

This project was developed as part of a major internship at **Uplyx Solution**. The main goal is to create a robust system that:

- 🧲 Dynamically fetches stock data from the internet
- 🔍 Analyzes patterns using technical indicators
- 🧠 Trains LSTM models to understand price movement
- 📈 Predicts and visualizes stock prices with Buy/Sell signals

---

## 🛠️ Tools & Libraries Used

| Category | Tools |
|---|---|
| **Data Collection** | `yfinance`, `pandas`, `numpy` |
| **Modeling** | `TensorFlow`, `Keras`, `LSTM` |
| **Indicators** | `ta`, `pandas_ta` *(optional)* |
| **Visualization** | `matplotlib`, `seaborn`, `plotly` *(in notebooks)* |
| **Web Deployment** | `Streamlit`, `.streamlit/config.toml` |
| **Notebook Experiments** | `Jupyter Notebook`, `ipywidgets` *(optional)* |

---

## 📂 Folder Structure
LSTM_Stock_Price_Prediction/
├── app.py ← Streamlit app
├── scripts/ ← Data fetching, preprocessing, modeling code
│ ├── data_collector.py
│ ├── preprocessing.py
│ ├── lstm_model.py
│ └── evaluate.py
├── notebooks/ ← Jupyter notebooks
│ ├── EDA.ipynb
│ ├── Indicators_Analysis.ipynb
│ ├── Model_Training_Experiments.ipynb
│ └── Live_Stock_Testing.ipynb
├── results/ ← Graphs, results, predictions
├── .streamlit/
│ └── config.toml ← Streamlit theme config
├── requirements.txt ← Python dependencies

---

## 🔍 Features

- 📈 **LSTM Neural Network** for sequence learning
- 📊 **MACD & RSI** for trend momentum and signals
- 🧠 **Buy/Sell Suggestions** using combined logic
- 🧪 Interactive **notebooks** for EDA, training, and testing
- 🖥️ **Streamlit Web App** for live predictions

---

## ⚙️ How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/your-username/LSTM_Stock_Price_Prediction.git
cd LSTM_Stock_Price_Prediction
pip install -r requirements.txt
streamlit run app.py
📊 Sample Visuals
Closing Price Trends

MACD & Signal Line with Buy/Sell Cues

RSI Zones (Overbought / Oversold)

LSTM Predicted vs Actual Stock Price


🧪 Run in Notebook
Navigate to the notebooks/ directory and run:

EDA.ipynb → Understand stock data and trends

Indicators_Analysis.ipynb → Study MACD & RSI quality

Model_Training_Experiments.ipynb → Test model configs

Live_Stock_Testing.ipynb → Predict any stock dynamically

✅ Future Improvements
🧠 Add GRU/Transformer models

📦 Add real-time stock alerts with email

📊 Integrate Bollinger Bands

🔐 Deploy with login authentication (Streamlit Secrets)
