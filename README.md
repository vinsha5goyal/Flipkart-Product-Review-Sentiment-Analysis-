

---

## 🛒 Flipkart Review Sentiment Analyzer

A machine learning web application that analyzes Flipkart product reviews and predicts whether they are **positive** or **negative**, using **Naive Bayes** and **Random Forest** classifiers.

---

### 📌 Features

* 🔠 **Text Preprocessing** with stemming and stopword removal
* 🧠 **ML Models**: Naive Bayes and Random Forest
* ✨ **TF-IDF Vectorization** of product reviews
* 🌥️ **Word Clouds** for visual insights
* 📊 **Sentiment Distribution** using bar and pie charts
* 📉 **Confusion Matrices** for both classifiers
* 🟩 **Top Words Visualization** by sentiment
* 📈 Model Accuracy and Performance Summary
* 🖥️ Streamlit web interface for ease of use

---

### 📁 Project Structure

```
flipkart-sentiment-analyzer/
│
├── app.py               # Main Streamlit application
├── flipkart_data.csv    # Dataset of product reviews and ratings
├── analysis.py          # (Optional) Data visualizations and exploration
├── index.html           # Landing page to launch Streamlit app
├── style.css            # Custom CSS styling for landing page
└── README.md            # Project overview and instructions
```

---

### 🚀 How to Run the Project

#### ✅ Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/flipkart-sentiment-analyzer.git
cd flipkart-sentiment-analyzer
```

#### ✅ Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Or install manually:**

```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn wordcloud streamlit
```

> Also run this once to download NLTK stopwords:

```python
import nltk
nltk.download('stopwords')
```

#### ✅ Step 3: Launch the Streamlit App

```bash
streamlit run app.py
```

The app will run locally at: `http://localhost:8501`

---

## 📸 Screenshots

### 🐾 Before_Adding_Input

![Before_Adding_Input](project_screenshots/Before_Adding_Input)

### 🧪 Dataset Details

* Source: Flipkart Product Reviews (sample dataset)
* Columns:

  * `review`: Text review by the user
  * `rating`: Numeric star rating (1 to 5)

#### 💡 Sentiment Mapping:

* `Positive` → Ratings **4 and 5**
* `Negative` → Ratings **1, 2, and 3**

---

### 📊 Output Visualizations

* ✅ Word Cloud for Positive Reviews
* ❌ Word Cloud for Negative Reviews
* 📈 Sentiment distribution (bar and pie)
* 📉 Confusion matrix heatmaps for both models
* 📌 Top words contributing to predictions

---

### 📌 Accuracy Summary

* **Naive Bayes Accuracy**: \~91%
* **Random Forest Accuracy**: \~90%
  *(Exact values will vary based on your dataset split.)*

---

### 🌐 Web Interface 

To launch from a landing page:

* Open `index.html` in your browser.
* Click **“Launch App”** to start the Streamlit interface (requires `streamlit run app.py` to be active).

---

### 🛠️ Future Improvements

* Deploy on Streamlit Cloud or Heroku
* Add review input and live sentiment prediction
* Multi-language review support
* Better HTML and CSS interface styling

---

### 📜 License

This project is open-source and free to use under the [Apache 2.0 License](LICENSE).

---

### 🙌 Acknowledgements

* Flipkart for sample product review data
* NLTK, Scikit-learn, and Streamlit communities
* Inspired by real-world e-commerce review analysis systems

---

