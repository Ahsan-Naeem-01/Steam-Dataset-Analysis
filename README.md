# 🎮 Steam Games Positive Ratings Predictor

This project is a fully interactive Streamlit web app that predicts the number of **positive user ratings** for games listed on the Steam store. It combines **exploratory data analysis**, **machine learning modeling**, and **real-time prediction**, all wrapped in an intuitive UI.

---

## 📌 Overview

We analyze the [Steam Store Games dataset](https://www.kaggle.com/datasets/nikdavis/steam-store-games) to understand key trends and build a regression model that predicts how positively a game might be received. Users can explore data insights, view model performance, and try live predictions based on game attributes.

---

## 🚀 Features

- 📊 **Data Exploration**  
  Explore trends across genres, age requirements, pricing, and user engagement.

- 🧹 **Data Preprocessing**  
  Cleaned missing values, handled outliers, simplified multi-label columns, and engineered new features.

- 🌲 **Modeling**  
  Trained a **Random Forest Regressor** to predict `positive_ratings`, using numeric and derived game features.

- 📈 **Performance Metrics**  
  Evaluated using **Mean Absolute Error**, **R² Score**, and **Accuracy** (on a custom threshold).

- 🔮 **Real-Time Prediction**  
  Users can input game details to predict how well it might be rated on Steam.

---

## 🧠 Tech Stack

- Python
- Streamlit
- Pandas, NumPy
- Scikit-learn
- Seaborn, Matplotlib
- Joblib

---

## 🗂️ Dataset

- Source: [Steam Store Games - Kaggle](https://www.kaggle.com/datasets/nikdavis/steam-store-games)
- Size: ~27,000 games with 18 attributes including genre, price, playtime, ratings, and more.

---

## 📁 Folder Structure
├── app.py # Main Streamlit app  
├── steam.csv # Cleaned dataset (optional)  
├── requirements.txt # Required libraries  
└── README.md # Project documentation

---

## 💻 How to Run

1. Clone the repo:

```bash
git clone https://github.com/your-username/steam-game-rating-predictor.git
cd steam-game-rating-predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Launch the app:
```bash
streamlit run app.py
```

