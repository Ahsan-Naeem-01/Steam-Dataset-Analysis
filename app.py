import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, accuracy_score, mean_absolute_error, confusion_matrix, classification_report

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import os


st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            width: 100px;
        }
        [data-testid="stSidebar"] > div:first-child {
            width: 300px;
        }
    </style>
    """, unsafe_allow_html=True)
# st.set_page_config(layout='wide')
st.title("🎮 Steam Games Positive Rating Prediction")

@st.cache_data
def load_data():
    df = pd.read_csv("steam.csv")
    return df
df = load_data()

df = load_data()

menu = st.sidebar.radio("Sections", ["Introduction","EDA","Model & Predict","Conclusion"])

if menu == "Introduction":
    st.header("📌 Overview")
    st.markdown("""
    This interactive web app explores and models the **Steam Store Games** dataset from Kaggle, aiming to predict the number of **positive user ratings** a game receives based on its features.

    🎮 **Why this matters:** With thousands of games available, understanding what influences a game's popularity can help developers, marketers, and analysts make data-driven decisions.

    🛠️ **What this app offers:**
    - A full **exploratory data analysis (EDA)** of the dataset including genre trends, developer stats, and rating distributions.
    - A machine learning model (**Random Forest Regressor**) trained to predict positive ratings using game features like price, playtime, and achievements.
    - A **real-time prediction interface** where users can input game details and receive an estimated number of positive ratings.

    📊 This project demonstrates a complete data science pipeline — from cleaning and visualizing the data to deploying a live model.
    """)


elif menu=="EDA":
    st.title("📊 Steam Dataset - Exploratory Data Analysis")

    # Define numerical columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    tab1, tab2 = st.tabs(["📊 Data Summary & Distribution", "📈 Insights & Trends"])


    with tab1:
        st.subheader("Summary Statistics")
        st.write(df.describe())

        st.subheader("Mode of Columns")
        modes = df.mode().iloc[0]
        types = df.dtypes

        mode_df = pd.DataFrame({
            'Column': modes.index.astype(str),
            'Mode': modes.values.astype(str),
        })
        st.dataframe(mode_df)

        st.subheader("Missing Value Analysis")
        missing_df = df.isnull().sum().reset_index()
        missing_df.columns = ['Column', 'Missing Values']
        missing_df['Missing Values'] = missing_df['Missing Values'].astype(int)

        st.dataframe(missing_df)
        # Data types and unique value counts
        st.subheader("Data Types and Unique Counts")
        types_uniques_df = pd.DataFrame({
            'Data Type': df.dtypes.astype(str),
            'Unique Values': df.nunique()
        }).reset_index().rename(columns={'index': 'Column'})

        st.dataframe(types_uniques_df)
        df['total_ratings'] = df['positive_ratings'] + df['negative_ratings']
        df = df[df['total_ratings'] > 0]
        df['pos_rate'] = df['positive_ratings'] / df['total_ratings']

        st.subheader("Histograms (Feature Distribution)")
        # Histograms
        st.subheader("Distribution of Positive Rating Ratio")
        fig2, ax2 = plt.subplots()
        sns.histplot(df['pos_rate'], bins=50, kde=True, ax=ax2)
        st.pyplot(fig2)

        st.subheader("Distribution of Price")
        fig2, ax2 = plt.subplots()
        sns.histplot(df['price'], bins=50, kde=True, ax=ax2)
        st.pyplot(fig2)

        st.subheader("Distribution of Average Playtime")
        fig2, ax2 = plt.subplots()
        sns.histplot(df['average_playtime'], bins=50, kde=True, ax=ax2)
        st.pyplot(fig2)
        st.subheader("Outlier Detection (IQR Counts)")
        def iqr_outliers(s):
            Q1, Q3 = s.quantile([0.25,0.75])
            return ((s < Q1 - 1.5*(Q3-Q1)) | (s > Q3 + 1.5*(Q3-Q1))).sum()
        outlier_counts = {col: iqr_outliers(df[col]) for col in num_cols}
        st.write(outlier_counts)

        st.subheader("Correlation Analysis")
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)


        if st.checkbox("Show Pairplot (May Take Time)"):
            st.subheader("Pairwise Relationships")
            st.write("This may take time due to high memory usage.")
            fig = sns.pairplot(
                df[num_cols].dropna().sample(min(500, len(df))),
                corner=True,
            )    
            st.pyplot(fig)


    with tab2:
        st.title("🎮 Steam Game Insights & Trends")

        st.subheader("Release Year Distribution")
        if 'release_date' in df.columns:
            df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
            release_counts = df['release_year'].value_counts().sort_index()
            st.line_chart(release_counts)

        st.subheader("Free vs Paid Games")
        if 'price' in df.columns:
            df['is_free'] = df['price'] == 0
            free_counts = df['is_free'].value_counts()
            free_counts.index = ['Paid' if not x else 'Free' for x in free_counts.index]
            st.bar_chart(free_counts)


        st.subheader("Top Developers ")
        if 'developer' in df.columns and 'positive_ratings' in df.columns:
            top_devs = (
                df.groupby('developer')['positive_ratings']
                .sum()
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
            )
            fig = px.bar(
                top_devs,
                x='developer',
                y='positive_ratings',
                title='Top 10 Developers With Most Positive Ratings',
                labels={'developer': 'Developer', 'positive_ratings': 'Positive Ratings'},
            )
            fig.update_layout(xaxis=dict(categoryorder='total descending'), 
                            xaxis_tickangle=-90)
            st.plotly_chart(fig, use_container_width=True)
        


        st.subheader("Most Common Genres")
        if 'genres' in df.columns:
            genre_counter = Counter()
            for entry in df['genres'].dropna():
                for genre in entry.split(';'):
                    genre_counter[genre.strip()] += 1

            genre_df = pd.DataFrame(genre_counter.items(), columns=['Genre', 'Count'])
            genre_df = genre_df.sort_values(by='Count', ascending=False).head(10)

            fig = px.bar(
                genre_df,
                x='Genre',
                y='Count',
                title='Top 10 Most Common Genres',
                labels={'Genre': 'Genre', 'Count': 'Frequency'},
            )
            fig.update_layout(xaxis=dict(categoryorder='total descending'))
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Most Common Categories")
        if 'categories' in df.columns:
            category_counter = Counter()
            for entry in df['categories'].dropna():
                for cat in entry.split(';'):
                    category_counter[cat.strip()] += 1

            category_df = pd.DataFrame(category_counter.items(), columns=['Category', 'Count'])
            category_df = category_df.sort_values(by='Count', ascending=False).head(10)

            fig = px.bar(
                category_df,
                x='Category',
                y='Count',
                title='Top 10 Most Common Categories',
                labels={'Category': 'Category', 'Count': 'Frequency'},
            )
            fig.update_layout(xaxis=dict(categoryorder='total descending'))
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Top 10 Games with the Most Negative Ratings")
        if 'name' in df.columns and 'negative_ratings' in df.columns:
            df_sorted = df.sort_values(by="negative_ratings", ascending=False)
            top_10_negative = df_sorted[['name', 'negative_ratings']].head(10)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=top_10_negative['name'],
                y=top_10_negative['negative_ratings'],
                # marker_color='crimson',
                name='Negative Ratings'
            ))

            fig.update_layout(
                title='Top 10 Games with Most Negative Ratings',
                xaxis_title='Game Title',
                yaxis_title='Number of Negative Ratings',
                xaxis_tickangle=-45
            )

            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Top 10 Games with the Most Positive Ratings")
        if 'name' in df.columns and 'positive_ratings' in df.columns:
            df_sorted = df.sort_values(by="positive_ratings", ascending=False)
            top_10_negative = df_sorted[['name', 'positive_ratings']].head(10)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=top_10_negative['name'],
                y=top_10_negative['positive_ratings'],
                # marker_color='crimson',
                name='positive Ratings'
            ))

            fig.update_layout(
                title='Top 10 Games with Most Positive Ratings',
                xaxis_title='Game Title',
                yaxis_title='Number of Positive Ratings',
                xaxis_tickangle=-45
            )

            st.plotly_chart(fig, use_container_width=True)


elif menu == "Model & Predict":
    st.header("Model Training & Prediction")
    st.markdown("""
    In this section, we use a **Random Forest Regressor** — a powerful ensemble model that builds multiple decision trees and averages their outputs to produce reliable predictions.
    It’s well-suited for this regression task because:
    - It handles non-linear relationships well
    - It’s robust to outliers
    - It requires little preprocessing

    We use it to predict the number of **positive user ratings** based on selected game features.
    """)
    model_path = "rf_model.pkl"
    model_trained_now = False
    model_loaded = False
    rf_model = None

    # Load and preprocess
    df = load_data()
    df = df.drop(['appid', 'categories', 'steamspy_tags', 'platforms'], axis=1)
    df["genres"] = df["genres"].str.split(";", n=1, expand=True)[0]
    df["total_ratings"] = df["positive_ratings"] + df["negative_ratings"]

    # Define features
    df_sample = df.sample(n=5000, random_state=42)
    X_sample = df_sample[['required_age', 'achievements', 'average_playtime', 'price', 'total_ratings']]
    y_sample = df_sample['positive_ratings']
    X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)

    if os.path.exists(model_path):
        rf_model = joblib.load(model_path)
        model_loaded = True


    if st.button("🚀 Train Model"):
        rf_model = make_pipeline(StandardScaler(), RandomForestRegressor(random_state=42))
        rf_model.fit(X_train, y_train)
        joblib.dump(rf_model, model_path)
        st.success("🎉 Model trained successfully.")
        model_loaded = True
        model_trained_now = True

    if model_loaded:
        y_pred_rf = rf_model.predict(X_test)

        mae_rf = mean_absolute_error(y_test, y_pred_rf)
        r2_rf = r2_score(y_test, y_pred_rf)
        threshold = df['positive_ratings'].median() 
        y_pred_class_rf = (y_pred_rf > threshold).astype(int)
        y_test_class_rf = (y_test > threshold).astype(int)
        ac_rf = accuracy_score(y_test_class_rf, y_pred_class_rf)
        conf_matrix_rf = confusion_matrix(y_test_class_rf, y_pred_class_rf)

        st.subheader("📊 Model Performance Metrics")
        st.metric("Mean Absolute Error (MAE)", f"{mae_rf:.2f}")
        st.metric("R² Score", f"{r2_rf:.2f}")
        st.metric(f"Accuracy (Threshold > {df['positive_ratings'].median() })", f"{ac_rf:.2f}")

        st.subheader("📉 Confusion Matrix")
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(conf_matrix_rf, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        ax_cm.set_xlabel('Predicted')
        ax_cm.set_ylabel('Actual')
        st.pyplot(fig_cm)

        st.subheader("Classification Report")
        report = classification_report(y_test_class_rf, y_pred_class_rf, target_names=['Not Popular', 'Popular'], output_dict= True)
        report_df = pd.DataFrame(report).transpose()

        st.dataframe(report_df.style.format("{:.2f}"))

        st.subheader("📈 Actual vs Predicted Positive Ratings")
        fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
        ax_scatter.scatter(y_test, y_pred_rf, alpha=0.6)
        ax_scatter.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax_scatter.set_xlabel("Actual Ratings")
        ax_scatter.set_ylabel("Predicted Ratings")
        ax_scatter.set_title("Prediction vs Reality")
        st.pyplot(fig_scatter)

        st.subheader("🎮 Try Live Prediction")
        with st.form("prediction_form"):
            st.markdown("Fill in the game details and click **Predict** to estimate the number of positive ratings.")
            col1, col2, col3 = st.columns(3)

            with col1:
                required_age_input = st.number_input("Required Age", min_value=0, max_value=100, value=18)
                achievements_input = st.number_input("Achievements", min_value=0, max_value=1000, value=10)

            with col2:
                average_playtime_input = st.number_input("Average Playtime (mins)", min_value=0, max_value=100000, value=120)
                price_input = st.number_input("Price ($)", min_value=0.0, max_value=100.0, value=19.99, step=0.01)

            with col3:
                total_ratings_input = st.number_input("Total Ratings", min_value=0, max_value=1000000, value=500)

            submitted = st.form_submit_button("🔮 Predict")

        if submitted:
            user_data = pd.DataFrame([{
                'required_age': required_age_input,
                'achievements': achievements_input,
                'average_playtime': average_playtime_input,
                'price': price_input,
                'total_ratings': total_ratings_input
            }])
            user_prediction = rf_model.predict(user_data)[0]
            st.success(f"🎯 **Predicted Positive Ratings:** {int(user_prediction)}")


elif menu == "Conclusion":
    st.header("✅ Conclusion")
    st.markdown("""
    In this project, we analyzed and modeled the **Steam Store Games** dataset to predict the number of **positive ratings** a game receives. Here's a comprehensive summary of our work:

    🔍 **Data Overview & Preprocessing**
    - The dataset contained **27,075 rows and 18 columns**, including features like `price`, `average_playtime`, `genres`, `achievements`, `required_age`, and more.
    - We created a new column `total_ratings` by summing `positive_ratings` and `negative_ratings`, and computed `pos_rate` to understand positive rating proportion.
    - Missing value analysis, datatype checks, and unique value inspection were performed to understand the structure of the data.

    📊 **Exploratory Data Analysis (EDA)**
    - We visualized the distribution of key numerical features such as `positive_ratings`, `price`, and `average_playtime` using histograms.
    - Outliers were detected using the IQR method. Notably, the features with the most outliers included:
        - `average_playtime`: 6,170 outliers
        - `positive_ratings`: 4,286 outliers
        - `negative_ratings`: 3,957 outliers
        - `price`: 1,975 outliers
    - A heatmap of correlations was generated to understand relationships between numerical variables.
    - Pairplots were optionally displayed for in-depth pairwise exploration.
    - Under insights, we analyzed:
        - **Release year trends** for games.
        - Distribution of **Free vs Paid** games.
        - Most popular **developers**, **genres**, and **categories** by frequency and rating counts.
        - Top 10 games with the **most positive** and **most negative** ratings were visualized using interactive bar plots.

    🤖 **Model Building & Prediction**
    - A **Random Forest Regressor** model was trained using the following features:
        - `required_age`, `achievements`, `average_playtime`, `price`, `total_ratings`
    - We sampled 5,000 records from the dataset for faster model training and encoded categorical variables using **OneHotEncoder**.
    - Features were standardized using `StandardScaler` in a pipeline.
    - The model was evaluated on a test set (20% split) and achieved the following performance:
        - 📈 **Mean Absolute Error (MAE):** 168.24
        - 📊 **R² Score:** 0.95
        - ✅ **Accuracy** (based on predicting games with >24.0 positive ratings): 0.95
    - A confusion matrix and a scatterplot of actual vs. predicted ratings were provided for evaluation.

    🧠 **Interactive Prediction**
    - An input form was built where users can enter game parameters like required age, achievements, average playtime, price, and total ratings.
    - The trained model makes a **real-time prediction** of expected positive ratings for the given game input.

    ✅ **Final Thoughts**
    - This app combines thorough **EDA**, robust **feature engineering**, solid **machine learning modeling**, and an intuitive **user interface**.
    - It successfully demonstrates the end-to-end data science workflow — from data understanding and visualization to model deployment and interactivity.
    """)

