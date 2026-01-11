# YouTube Comment Sentiment Analysis

## 📌 Project Overview
This project performs **sentiment analysis on YouTube comments** using **web scraping, Natural Language Processing (NLP), Machine Learning, and Deep Learning techniques**.  
The goal is to classify user comments into **Positive, Negative, and Neutral sentiments** and analyze model performance.

This project demonstrates an **end-to-end data science pipeline**, starting from data collection to model evaluation.

---

## 🎯 Objectives
- Scrape real-time YouTube comments from a public video
- Convert unstructured text data into a structured dataset
- Preprocess and clean textual data
- Apply machine learning and deep learning models for sentiment classification
- Evaluate models using standard performance metrics

---

## 🛠️ Technologies & Libraries Used
- **Python**
- **Selenium** – Web scraping dynamic YouTube comments
- **Pandas, NumPy** – Data manipulation and analysis
- **NLTK** – Text preprocessing (tokenization, stopword removal)
- **Scikit-learn** – TF-IDF vectorization and ML models
- **TensorFlow / Keras** – Deep learning (LSTM model)
- **Matplotlib / Seaborn** – Visualization

---

## 📊 Dataset
- **Source:** YouTube comments scraped using Selenium
- **File Name:** `Comments_youtube.csv`
- **Description:** Contains real user comments extracted from YouTube videos for sentiment analysis

---

## 🔍 Methodology

### 1️⃣ Data Collection
- Used **Selenium** to automate browser actions
- Handled dynamically loaded YouTube comments
- Scrolled the page to load more comments
- Extracted comment text and saved it into a CSV file

**Benefit:**  
Allows collection of real-world, unbiased user opinions from dynamic websites.

---

### 2️⃣ Data Preprocessing
- Converted text to lowercase
- Removed punctuation and special characters
- Removed stopwords
- Tokenized text into words

**Benefit:**  
Improves data quality and enhances model performance by reducing noise.

---

### 3️⃣ Feature Extraction
- Applied **TF-IDF Vectorization** for machine learning models
- Used **tokenization and padding** for deep learning models

**Benefit:**  
Transforms text into numerical form suitable for ML and DL algorithms.

---

### 4️⃣ Model Implementation

#### 🔹 Machine Learning Models
- Logistic Regression
- Naive Bayes
- Support Vector Machine (SVM)

**Benefits:**
- Fast training
- Easy to interpret
- Strong baseline performance

#### 🔹 Deep Learning Model
- LSTM (Long Short-Term Memory) with embedding layer

**Benefits:**
- Captures contextual and sequential information
- Performs better on complex sentences
- Represents modern NLP techniques

---

### 5️⃣ Model Evaluation
The models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

**Benefit:**  
Provides a fair and detailed comparison of model performance beyond accuracy alone.

---

## ✅ Results
- Machine learning models provided reliable baseline results
- LSTM model showed improved understanding of context and sentiment
- Performance comparison helped identify the most effective approach

---

## 📁 Project Structure

├── Asg4_ANIKET_MITTAL_M24DS001.ipynb
├── Comments_youtube.csv
├── README.md

### 1️⃣ Install required libraries
```bash
pip install selenium pandas numpy nltk scikit-learn tensorflow matplotlib

jupyter notebook HR_Analysis_youtube_video.ipynb

