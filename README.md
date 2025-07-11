# Flipkart-Product-Review-Sentiment-Analysis-
Sentiment analysis of Flipkart product reviews using NLP and ML (Naive Bayes,  Random Forest) with up to 91% accuracy. Includes GUI for live prediction via Streamlit
Here’s a professional `README.md` file you can include in your GitHub repository for the Flipkart Review Sentiment Analysis project. It explains the project's goal, how it works, and how to run it:

---

# 📊 Flipkart Product Review Sentiment Analysis

This project analyzes Flipkart product reviews using Natural Language Processing (NLP) and Machine Learning techniques to classify sentiments as **positive** or **negative**. It compares two popular classifiers: **Naive Bayes** and **Random Forest**, visualizes key insights from the data, and highlights important predictive features.

## 📁 Dataset

* `flipkart_data.csv`: Contains user reviews and ratings scraped from Flipkart.
* Columns used: `review`, `rating`

## 🔧 Tech Stack

* Python (pandas, NumPy, re, nltk)
* Machine Learning: scikit-learn
* Visualizations: matplotlib, seaborn, WordCloud
* NLP: stopword removal, stemming, TF-IDF vectorization

---

## 🚀 Features

* Clean and preprocess raw Flipkart reviews
* Generate Word Clouds for positive and negative sentiments
* Visualize sentiment distribution using bar and pie charts
* Train two models: Naive Bayes & Random Forest
* Display confusion matrices as heatmaps
* Show most indicative positive/negative words for Naive Bayes

---

## 🧠 ML Models Used

| Model            | Accuracy |
| ---------------- | -------- |
| ✅ Naive Bayes    | \~91%    |
| 🌲 Random Forest | \~93%    |

(Note: Accuracy may vary slightly depending on data split and parameters.)

---

## 📌 Visualizations

* ✅ WordClouds (Positive & Negative)
* 📊 Bar Chart: Sentiment Count
* 🥧 Pie Chart: Sentiment Proportions
* 📉 Heatmaps: Confusion Matrices
* 🔤 Top 20 Words (Naive Bayes): Important for each sentiment

---

## 🧪 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/flipkart-sentiment-analysis.git
cd flipkart-sentiment-analysis
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the script

```bash
python sentiment_analysis.py
```

Make sure the `flipkart_data.csv` file is in the same directory as the script.

---

## 📁 Project Structure

```
├── flipkart_data.csv
├── sentiment_analysis.py
├── README.md
└── requirements.txt
```

---

## ✅ Output

At the end of the script, you'll see:

* Accuracy results of both models
* Word clouds
* Bar and pie charts
* Confusion matrices
* Top 20 predictive words for each sentiment

---

## 📌 Requirements

You can create a `requirements.txt` like this:

```txt
pandas
numpy
matplotlib
seaborn
nltk
scikit-learn
wordcloud
```

---

## ✍️ Author

Vinsha Goyal
[LinkedIn](#) | [GitHub](https://github.com/your-username)

---


