import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

data = {
    'text' : ["Hello", "Hi", "Hey", "Bye", "See you later", "Thank you", "Thank you so much"],
    'intent' : ["greetings", "greetings", "greetings", "goodbye", "goodbye", "thanks", "thanks"]
}

df = pd.DataFrame(data)

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['intent'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

clf = MultinomialNB()
clf.fit(X_train_vectorized, y_train)

accuracy = clf.score(X_test_vectorized, y_test)
print(f"Accuracy: {accuracy}")

def predict_intent(text):
    text_vectorized = vectorizer.transform([text])
    prediction = clf.predict(text_vectorized)
    return prediction[0]

print(predict_intent("Hello"))
