# Restaurant Sentiment Analysis 🔍🍽️

End-to-end NLP pipeline classifying restaurant reviews as **Positive** or **Negative** using TF-IDF vectorization and multiple ML classifiers. Achieved **~75% accuracy** through text preprocessing and model comparison.

## ✨ Features
- **Text preprocessing**: Regex cleaning, NLTK lemmatization, custom stopwords (kept 'not')
- **Feature extraction**: TF-IDF with bigrams (3000 features)
- **Model comparison**: Multinomial Naive Bayes (alpha=0.1) vs Logistic Regression
- **Evaluation**: Confusion matrix, accuracy, precision/recall/F1 scores

## 📊 Results
| Model | Accuracy | Key Insight |
|-------|----------|-------------|
| Naive Bayes | 76% | Good baseline |
| Logistic Regression | **78%** | Better precision on negative reviews |

## 🛠️ Tech Stack
Python 3.9+ | scikit-learn | NLTK | pandas | numpy | re | Jupyter

## 🚀 Quick Start
```bash
# Clone repo
git clone https://github.com/krishnagupta1543/restaurant-sentiment-analysis

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook Restaurant_Sentiment_Analysis.ipynb
```

## 📈 Model Performance
```python
# Best model: Logistic Regression (max_iter=1000)
# Handles negation words correctly by keeping 'not' in preprocessing
```

## 💡 Learning Outcomes
- Regex patterns for text cleaning (`[^a-zA-Z] → ' '`)
- Custom stopwords handling for sentiment analysis
- TF-IDF bigrams improve context understanding
- Model selection through accuracy + confusion matrix

## 📂 Dataset
- **Source**: Restaurant reviews TSV (tab-separated)
- **Columns**: Review text + Label (0=Negative, 1=Positive)

## 📧 Contact

**Krishna Gupta**  
Email: guptak143600@gmail.com 
GitHub: https://github.com/Krishna28Gupta
