import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load Data
print("Loading training, validation, and test data...")
train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')
test = pd.read_csv('test.csv')

# 2. TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_train = tfidf.fit_transform(train['message'])
X_val = tfidf.transform(val['message'])
X_test = tfidf.transform(test['message'])

y_train = train['new_label']
y_val = val['new_label']
y_test = test['new_label']

# 3. Train Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
print("Model training complete")

# 4. Validation Evaluation
val_preds = nb_model.predict(X_val)

print("\n--- Validation Performance ---")
print(classification_report(y_val, val_preds))

cm = confusion_matrix(y_val, val_preds, labels=nb_model.classes_)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=nb_model.classes_,
            yticklabels=nb_model.classes_)
plt.title('DOTA 2 Classification: Actual vs. Predicted')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.savefig('nbc_validation_confusion.png', dpi=150, bbox_inches='tight')
print("Validation confusion matrix saved to nbc_validation_confusion.png")

# 5. Test Set Evaluation
test_preds = nb_model.predict(X_test)

print("\n--- Test Set Classification Report ---")
print(classification_report(y_test, test_preds))

cm_test = confusion_matrix(y_test, test_preds, labels=nb_model.classes_)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens',
            xticklabels=nb_model.classes_,
            yticklabels=nb_model.classes_)
plt.title('NBC Final Performance: Test Set')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.savefig('nbc_test_confusion.png', dpi=150, bbox_inches='tight')
print("Test confusion matrix saved to nbc_test_confusion.png")