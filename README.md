# 📊 End-to-End ANN Customer Churn Prediction

This project is an **End-to-End Artificial Neural Network (ANN) based Customer Churn Prediction System** that predicts whether a bank customer is likely to leave (churn) or stay with the company. The project covers the full ML lifecycle:

- ✅ Data Cleaning  
- ✅ Exploratory Data Analysis (EDA) & Visualization  
- ✅ Feature Engineering  
- ✅ ANN Model Training  
- ✅ Model Evaluation  
- ✅ Deployment using **Streamlit Web App**

---

## 🎯 Project Objective

The main objective of this project is to help banks **identify potential churners in advance**, so they can:

- Offer personalized retention strategies  
- Improve customer experience  
- Reduce financial losses  
- Design targeted loyalty programs  

This makes the system valuable for real-world business decision-making.

---

## 📁 Dataset Description

The dataset contains the following customer attributes:

| Feature | Description |
|--------|-------------|
| CreditScore | Customer credit rating |
| Geography | Country of the customer (France, Germany, Spain) |
| Gender | Male or Female |
| Age | Customer age |
| Tenure | Years associated with the bank |
| Balance | Bank account balance |
| NumOfProducts | Number of bank products used |
| HasCrCard | Whether customer owns a credit card (0/1) |
| IsActiveMember | Active membership status (0/1) |
| EstimatedSalary | Customer’s salary |
| Exited (Target) | 1 = Churn, 0 = Not Churn |

---

## 📊 Exploratory Data Analysis & Data Visualization (EDA)

Before model training, detailed **data visualization and exploratory analysis** were performed to understand patterns and relationships in the data.

### Visualizations created:

- **Churn Distribution**
  - Showed whether the dataset was balanced or imbalanced  

- **Geography vs Churn**
  - Identified that customers from **Germany had higher churn rates**  

- **Gender vs Churn**
  - Compared churn behavior between Male and Female customers  

- **Age Distribution**
  - Revealed that older customers were more likely to churn  

- **Balance vs Churn**
  - Customers with higher balance showed higher churn tendency  

- **Correlation Heatmap**
  - Helped identify relationships among numerical features  

### Key Insights from Visualization:
- Higher balance → higher churn probability  
- Older customers churn more frequently  
- Geography significantly impacts churn  
- Active members are less likely to churn  

Tools used for visualization:


## 🧠 Model Architecture (ANN)

The model is built using **TensorFlow / Keras** with the following architecture:

- Input Layer: 11 features  
- Hidden Layer 1: 64 neurons + ReLU  
- Hidden Layer 2: 32 neurons + ReLU  
- Output Layer: 1 neuron + Sigmoid  
- Optimizer: Adam  
- Loss Function: Binary Crossentropy  

---

## 🔧 Data Preprocessing Steps

Before training, the following preprocessing steps were applied:

- Encoded categorical features:
  - `Geography` → OneHotEncoder  
  - `Gender` → LabelEncoder  
- Feature Scaling using **StandardScaler**
- Split dataset into:
  - Training set: 80%
  - Testing set: 20%

Saved preprocessing objects:
- `geo_encoder.pkl`
- `gen_encoder.pkl`
- `scalar.pkl`

---

## 📈 Model Performance

The trained model achieved:

- Accuracy: **~85%**
- Good generalization on unseen test data  
- Suitable for real-world predictions  

---

## 🌐 Deployment — Streamlit Web App

A user-friendly web interface was built using **Streamlit**, where users can input customer details and get instant churn prediction.

### Features of the Web App:
- Interactive sliders and input fields  
- Real-time prediction  
- Clear output:
  - ✅ “Customer is NOT likely to churn”
  - ⚠️ “Customer IS likely to churn”

## 🚀 Live Demo
🔗 Streamlit App: https://customerchurnprediction-uyzvbkpphnd6uxt8uxcegd.streamlit.app
