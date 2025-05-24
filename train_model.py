import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load dataset (make sure SMSSpamCollection file is in folder)
data = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['label', 'message'])
data['label_num'] = data.label.map({'ham':0, 'spam':1})

X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label_num'], test_size=0.2, random_state=42)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

# Save model and transformers
joblib.dump(clf, 'model.pkl')
joblib.dump(count_vect, 'count_vect.pkl')
joblib.dump(tfidf_transformer, 'tfidf_transformer.pkl')
