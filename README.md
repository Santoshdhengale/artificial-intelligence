echo "# artificial-intelligence" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/Santoshdhengale/artificial-intelligence.git
git push -u origin main

Overview

This project is a Machine Learning–based Spam Detection System that classifies text messages or emails as Spam or Not Spam.
It uses Natural Language Processing (NLP) techniques and a Naive Bayes classifier to detect spam based on word frequency patterns.
The model is deployed using Streamlit, providing an interactive web interface for real-time predictions.

Features

1)Classifies messages as Spam or Not Spam
2)Uses CountVectorizer for text vectorization
3)Trained with Multinomial Naive Bayes
4)Interactive Streamlit web app
5)Saves trained model and vectorizer using pickle

Machine Learning Pipeline

1)Data Loading – Spam dataset (spam.csv)
2)Data Cleaning
    Removed duplicate messages
    Label encoding (spam, ham)
3)Text Vectorization
    Converted text into numerical features using CountVectorizer
4)Model Training
    Multinomial Naive Bayes classifier
5)Prediction
    Real-time user input through Streamlit UI

Tech Stack
1.Python
2.Pandas
3.Scikit-learn
4.Streamlit
5.Pickle
6.NLP (Bag of Words)

How to run the Projecct 

1.Clone the repository
    git clone <your-repo-link>
    cd Spam-Detector

2.Create Virtual Environment
    python -m venv .venv
    .venv\Scripts\activate

3.Install Dpendencies
    pip install -r requirements.txt

4.Run streamlit app
    streamlit run Home.py


Model Used

Algorithm: Multinomial Naive Bayes

Vectorizer: CountVectorizer (with English stop words removed)

Future Improvements

1.Use TF-IDF Vectorizer for better accuracy
2.Add confidence score for predictions
3.Deploy on Streamlit Cloud
4.Try other models like Logistic Regression

Author

Atharv Dhane
Engineering Student | AI & Data Science
Machine Learning Enthusiast
