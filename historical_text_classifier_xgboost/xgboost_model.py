import json
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from nltk.corpus import stopwords
import nltk
import chardet
from os.path import join, dirname, abspath
import joblib

# Загрузка стоп-слов один раз
nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))


class Preprocessor:
    def __init__(self, stop_words):
        self.stop_words = stop_words

    def __call__(self, texts):
        return [self.preprocess_text(text) for text in texts]

    def preprocess_text(self, text):
        text = re.sub(r"[^а-яА-Яa-zA-Z0-9.,!?\s]", "", text)  # Удаление лишних символов
        text = text.lower()  # Приведение к нижнему регистру
        words = text.split()
        words = [word for word in words if word not in self.stop_words]  # Удаление стоп-слов
        return ' '.join(words)


class XGBoostTextClassifier:
    def __init__(self, dataset_path=None, model_path=None):
        self.dataset_path = dataset_path or join(dirname(abspath(__file__)), 'data', 'dataset.json')
        self.model_path = model_path or 'xgboost_model.pkl'
        self.pipeline = None
        self.preprocessor = Preprocessor(stop_words)

    def load_data(self):
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        texts = [item["text"] for item in data]
        labels = [1 if item["label"] == "historical_background" else 0 for item in data]
        return texts, labels

    def train(self):
        texts, labels = self.load_data()
        pipeline = Pipeline(steps=[
            ('cleaner', FunctionTransformer(self.preprocessor, validate=False)),
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('classifier', XGBClassifier(random_state=42))
        ])
        param_grid = {
            'tfidf__max_features': [5000, 10000],
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.01, 0.1],
            'classifier__max_depth': [3, 6, 9]
        }
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1, error_score='raise')
        grid_search.fit(texts, labels)
        self.pipeline = grid_search.best_estimator_
        self.save_model()

    def save_model(self):
        joblib.dump(self.pipeline, self.model_path)

    def load_model(self):
        self.pipeline = joblib.load(self.model_path)

    def predict(self, text):
        if not self.pipeline:
            self.load_model()
        preprocessed_text = self.preprocessor([text])
        return self.pipeline.predict(preprocessed_text)[0]

    def predict_proba(self, text):
        if not self.pipeline:
            self.load_model()
        preprocessed_text = self.preprocessor([text])
        return self.pipeline.predict_proba(preprocessed_text)[0]

    def evaluate(self):
        texts, labels = self.load_data()
        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
        y_pred = self.pipeline.predict(X_test)
        y_prob = self.pipeline.predict_proba(X_test)[:, 1]
        print("Best Parameters:", self.pipeline.named_steps['classifier'].get_params())
        print("Accuracy:", accuracy_score(y_test, y_pred))

        # Матрица ошибок
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Twaddle', 'Historical Background'],
                    yticklabels=['Twaddle', 'Historical Background'])
        plt.xlabel('Предсказанные значения')
        plt.ylabel('Фактические значения')
        plt.title('Матрица ошибок')
        plt.show()

        # ROC-кривая
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC кривая (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC-кривая')
        plt.legend(loc="lower right")
        plt.show()

    def extract_valuable_passages(self, input_text, threshold=0.5, min_length=10):
        paragraphs = input_text.split("\n")
        paragraphs = [p.strip() for p in paragraphs if len(p.strip().split()) >= min_length]
        paragraphs = [p for p in paragraphs if not any(kw in p.lower() for kw in stop_words)]
        preprocessed_paragraphs = self.preprocessor(paragraphs)

        # Проверка, что есть предобработанные абзацы
        if not preprocessed_paragraphs:
            return []

        # Фильтрация пустых абзацев после предобработки
        non_empty_preprocessed_paragraphs = [p for p in preprocessed_paragraphs if p]
        if not non_empty_preprocessed_paragraphs:
            return []

        paragraph_vectors = self.pipeline.named_steps['tfidf'].transform(non_empty_preprocessed_paragraphs)
        probabilities = self.pipeline.predict_proba(paragraph_vectors)[:, 1]
        results = [(para, prob) for para, prob in zip(paragraphs, probabilities) if prob >= threshold]
        return results

    def fetch_data_from_wikipedia(self, query, num_results=5):
        url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            results = data.get("query", {}).get("search", [])
            return [item["snippet"] for item in results][:num_results]
        else:
            return []

    def plot_feature_importance(self):
        if not self.pipeline:
            self.load_model()
        booster = self.pipeline.named_steps['classifier'].get_booster()
        importance = booster.get_score(importance_type='weight')
        importance = dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))
        feature_names = self.pipeline.named_steps['tfidf'].get_feature_names_out()
        feature_importances = np.array([importance.get(f, 0) for f in feature_names])
        top_n = 30
        sorted_indices = feature_importances.argsort()[::-1][:top_n]
        sorted_importances = feature_importances[sorted_indices]
        sorted_feature_names = feature_names[sorted_indices]
        plt.figure(figsize=(12, 6))
        plt.bar(range(top_n), sorted_importances, align="center")
        plt.xticks(range(top_n), sorted_feature_names, rotation=90)
        plt.xlabel("Слова")
        plt.ylabel("Важность")
        plt.title("Важность признаков (слова)")
        plt.show()