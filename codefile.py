import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -------------------------------
# Streamlit App Title
# -------------------------------
st.title("Wine Quality Prediction & Analysis")

# -------------------------------
# 1. Load dataset
# -------------------------------
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=";")
    return df

df = load_data()

# -------------------------------
# 2. Show dataset info
# -------------------------------
st.subheader("Dataset")
st.write(df.head())
st.write(f"Dataset Shape: {df.shape}")

if st.checkbox("Show missing values"):
    st.write(df.isnull().sum())

if st.checkbox("Show statistical description"):
    st.write(df.describe())

# -------------------------------
# 3. Data Visualization
# -------------------------------
st.subheader("Data Visualization")

# Correlation heatmap
if st.checkbox("Show Correlation Heatmap"):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    st.pyplot(plt)

# Histograms of features
if st.checkbox("Show Histograms of Features"):
    df.hist(figsize=(12, 10), bins=20)
    st.pyplot(plt)

# Pairplot
if st.checkbox("Show Pairplot"):
    st.write("This might take some time...")
    sns.pairplot(df.sample(100))  # Use a sample for speed
    st.pyplot(plt)

# -------------------------------
# 4. Model Training
# -------------------------------
X = df.drop("quality", axis=1)
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Accuracy")
st.write(f"Random Forest Accuracy: {accuracy:.2f}")

# -------------------------------
# 5. Predict wine quality
# -------------------------------
st.subheader("Predict Wine Quality")
st.write("Input features to predict wine quality:")

# Input sliders for features
input_data = {}
for column in X.columns:
    min_val = float(df[column].min())
    max_val = float(df[column].max())
    mean_val = float(df[column].mean())
    input_data[column] = st.slider(column, min_val, max_val, mean_val)

# Predict button
if st.button("Predict Quality"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Wine Quality: {prediction}")
