import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="ML Pipeline", layout="wide")
st.title("📊 Machine Learning Pipeline Application")

# Create models directory
if not os.path.exists("models"):
    os.makedirs("models")

# Sidebar for navigation
st.sidebar.header("Pipeline Steps")

# Step 1: Upload Dataset
st.sidebar.subheader("1. Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is None:
    st.info("👈 Please upload a CSV file to get started")
    st.stop()

# Load dataset
df = pd.read_csv(uploaded_file)
st.success("✅ Dataset loaded successfully!")

# Display original data
with st.expander("📋 View Original Data"):
    st.write(df.head())
    st.write(f"Shape: {df.shape}")

# Step 2: Data Cleaning
st.sidebar.subheader("2. Data Cleaning")
st.subheader("🧹 Data Cleaning")

col1, col2 = st.columns(2)

with col1:
    st.write("**Missing Values:**")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        st.write("✅ No missing values!")
    else:
        st.write(missing[missing > 0])

with col2:
    st.write("**Data Types:**")
    st.write(df.dtypes)

# Clean data
df_clean = df.copy()

# Handle missing values
missing_strategy = st.selectbox("Select missing value strategy:", 
                                ["Drop rows", "Fill with mean", "Fill with median"])

if df_clean.isnull().sum().sum() > 0:
    if missing_strategy == "Drop rows":
        df_clean = df_clean.dropna()
    elif missing_strategy == "Fill with mean":
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
    else:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())

# Remove duplicates
df_clean = df_clean.drop_duplicates()

# Encode categorical variables
categorical_cols = df_clean.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
    label_encoders[col] = le

st.success("✅ Data cleaning completed!")

with st.expander("📊 View Cleaned Data"):
    st.write(df_clean.head())
    st.write(f"Shape after cleaning: {df_clean.shape}")

# Step 3: Model Selection
st.sidebar.subheader("3. Model Configuration")
st.subheader("⚙️ Model Configuration")

# Select target variable
target_col = st.selectbox("Select target column:", df_clean.columns)

# Select model type
model_type = st.radio("Select model type:", ["Regression", "Classification"])

# Model selection
if model_type == "Regression":
    model_name = st.selectbox("Select regression model:", 
                              ["Linear Regression", "Random Forest Regressor"])
else:
    model_name = st.selectbox("Select classification model:", 
                              ["Logistic Regression", "Random Forest Classifier"])

# Prepare features and target
X = df_clean.drop(columns=[target_col])
y = df_clean[target_col]

# Train-test split
test_size = st.slider("Test size ratio:", 0.1, 0.5, 0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Train Model
st.sidebar.subheader("4. Training")

if st.button("🚀 Train Model", key="train_btn"):
    with st.spinner("Training model..."):
        # Initialize model
        if model_type == "Regression":
            if model_name == "Linear Regression":
                model = LinearRegression()
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            if model_name == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Save model
        model_path = f"models/{uploaded_file.name.split('.')[0]}_{model_type}_{model_name.replace(' ', '_')}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({'model': model, 'scaler': scaler, 'features': X.columns.tolist()}, f)
        
        st.success(f"✅ Model trained and saved to `{model_path}`")
        
        # Evaluate model
        st.subheader("📈 Model Performance")
        
        if model_type == "Regression":
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("RMSE", f"{rmse:.4f}")
            col2.metric("R² Score", f"{r2:.4f}")
            col3.metric("MSE", f"{mse:.4f}")
        else:
            accuracy = accuracy_score(y_test, y_pred)
            st.metric("Accuracy", f"{accuracy:.4f}")
            st.text(classification_report(y_test, y_pred))
        
        # Visualization
        st.subheader("📊 Predictions Visualization")
        
        # 2D Scatter plot
        if X_test.shape[1] >= 2:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # 2D Scatter
            scatter = axes[0].scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], 
                                     c=y_pred, cmap='viridis', s=50, alpha=0.6)
            axes[0].set_xlabel(f"Feature 0 (Scaled)")
            axes[0].set_ylabel(f"Feature 1 (Scaled)")
            axes[0].set_title("2D Prediction Visualization")
            plt.colorbar(scatter, ax=axes[0], label="Prediction")
            
            # Actual vs Predicted
            axes[1].scatter(y_test, y_pred, alpha=0.6, s=50)
            axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            axes[1].set_xlabel("Actual Values")
            axes[1].set_ylabel("Predicted Values")
            axes[1].set_title("Actual vs Predicted")
            
            st.pyplot(fig)
        
        # 3D Scatter plot
        if X_test.shape[1] >= 3:
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            
            scatter = ax.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], X_test_scaled[:, 2],
                               c=y_pred, cmap='plasma', s=50, alpha=0.6)
            ax.set_xlabel("Feature 0")
            ax.set_ylabel("Feature 1")
            ax.set_zlabel("Feature 2")
            ax.set_title("3D Prediction Visualization")
            
            st.pyplot(fig)
        
        # Interactive 3D plot with Plotly
        if X_test.shape[1] >= 3:
            fig_3d = go.Figure(data=[go.Scatter3d(
                x=X_test_scaled[:, 0],
                y=X_test_scaled[:, 1],
                z=X_test_scaled[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=y_pred,
                    colorscale='Viridis',
                    showscale=True
                )
            )])
            
            fig_3d.update_layout(
                title="Interactive 3D Prediction Visualization",
                scene=dict(
                    xaxis_title="Feature 0",
                    yaxis_title="Feature 1",
                    zaxis_title="Feature 2"
                ),
                width=1000,
                height=700
            )
            
            st.plotly_chart(fig_3d, use_container_width=True)
        
        st.info("💾 Model has been saved and will be overwritten on next training with the same name.")