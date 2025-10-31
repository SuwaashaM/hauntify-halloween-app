# hauntify_app.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
import streamlit as st

# -----------------------------
# Load or Train Model
# -----------------------------
MODEL_PATH = 'models/sentiment_model.pkl'
VECTORIZER_PATH = 'models/tfidf_vectorizer.pkl'

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    # Load dataset
    df = pd.read_csv(r"C:\Users\adminS\OneDrive\Data Science\hauntify_posts.csv")
    
    # Convert text to numbers
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['post'])
    y = df['label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Test model
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    # Save model and vectorizer
    os.makedirs('models', exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
else:
    # Load existing model
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)

# -----------------------------
# Helper Functions
# -----------------------------
def predict_spookiness(post):
    text_vec = vectorizer.transform([post])
    prediction = model.predict(text_vec)[0]
    score = model.predict_proba(text_vec)[0][1] * 100  # Probability of spookiness
    label = "🩸 Spooky!" if prediction == 1 else "🎃 Not spooky"
    return f"{label} (Spookiness Score: {score:.2f}%)", score

def recommend_costume(score):
    if score > 80:
        return "👻 Ghost, 💀 Skeleton, 🧛 Vampire, 🧙 Witch"
    elif score > 60:
        return "🧟 Zombie, 🧞 Genie, 🧝 Elf"
    elif score > 40:
        return "🧚 Fairy, 🐺 Werewolf, 🦹 Villain"
    else:
        return "🎩 Magician, 🐱 Cat, 🎃 Pumpkin"

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Hauntify 🎃", page_icon="👻", layout="wide")
st.title("Hauntify: Spookiness Predictor & Costume Recommender")

user_input = st.text_area("Enter your Halloween post or text:")

if st.button("Analyze"):
    if user_input.strip() != "":
        result, score = predict_spookiness(user_input)
        st.success(result)
        st.info(f"Recommended Costumes: {recommend_costume(score)}")
    else:
        st.warning("Please enter some text to analyze!")
