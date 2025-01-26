import json
import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import requests
import seaborn as sns
import matplotlib.pyplot as plt
import chardet
from os.path import join, dirname, abspath
import joblib


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, stop_words):
        self.stop_words = stop_words

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self.preprocess_text(text) for text in X]

    def preprocess_text(self, text):
        text = re.sub(r"[^а-яА-Яa-zA-Z0-9.,!?\s]", "", text)  # Удаление лишних символов
        text = text.lower()  # Приведение к нижнему регистру
        words = text.split()
        words = [word for word in words if word not in self.stop_words]  # Удаление стоп-слов
        return ' '.join(words)


class RandomForestTextClassifier:
    def __init__(self, dataset_path=None, model_path=None):
        self.dataset_path = dataset_path or join(dirname(abspath(__file__)), 'data', 'dataset.json')
        self.model_path = model_path or 'random_forest_model.pkl'
        self.pipeline = None
        self.stop_words = set(['акт', 'лист', 'приказ', 'распоряжение', '№'])

    def load_data(self):
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        texts = [item["text"] for item in data]
        labels = [1 if item["label"] == "historical_background" else 0 for item in data]
        return texts, labels

    def train(self):
        texts, labels = self.load_data()
        pipeline = Pipeline(steps=[
            ('cleaner', TextPreprocessor(stop_words=self.stop_words)),
            ('tfidf', TfidfVectorizer()),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
        pipeline.fit(X_train, y_train)
        self.pipeline = pipeline
        self.save_model()

    def save_model(self):
        joblib.dump(self.pipeline, self.model_path)

    def load_model(self):
        if not hasattr(self, 'pipeline'):
            try:
                self.pipeline = joblib.load(self.model_path)
            except FileNotFoundError:
                self.train()

    def predict(self, text):
        self.load_model()
        preprocessed_text = [text]  # Не нужно вызывать preprocess_text здесь
        return self.pipeline.predict(preprocessed_text)[0]

    def predict_proba(self, text):
        self.load_model()
        preprocessed_text = [text]  # Не нужно вызывать preprocess_text здесь
        return self.pipeline.predict_proba(preprocessed_text)[0]

    def evaluate_confusion_matrix(self):
        self.load_model()
        texts, labels = self.load_data()
        y_pred = self.pipeline.predict(texts)
        cm = confusion_matrix(labels, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Twaddle', 'Historical Background'],
                    yticklabels=['Twaddle', 'Historical Background'])
        plt.xlabel('Предсказанные значения')
        plt.ylabel('Фактические значения')
        plt.title('Матрица ошибок')
        plt.show()

    def evaluate_roc_curve(self):
        self.load_model()
        texts, labels = self.load_data()
        y_pred = self.pipeline.predict(texts)
        y_prob = self.pipeline.predict_proba(texts)[:, 1]
        fpr, tpr, thresholds = roc_curve(labels, y_prob)
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

    def feature_importance_plot(self):
        self.load_model()
        feature_importances = self.pipeline.named_steps['classifier'].feature_importances_
        feature_names = self.pipeline.named_steps['tfidf'].get_feature_names_out()
        sorted_indices = feature_importances.argsort()[::-1]
        sorted_importances = feature_importances[sorted_indices]
        sorted_feature_names = feature_names[sorted_indices]
        plt.figure(figsize=(12, 6))
        plt.bar(range(30), sorted_importances[:30], align="center")
        plt.xticks(range(30), sorted_feature_names[:30], rotation=90)
        plt.xlabel("Слова")
        plt.ylabel("Важность")
        plt.title("Важность признаков (слова)")
        plt.show()

    def extract_valuable_passages(self, input_text, threshold=0.5, min_length=10):
        paragraphs = input_text.split("\n")
        paragraphs = [p.strip() for p in paragraphs if len(p.strip().split()) >= min_length]
        paragraphs = [p for p in paragraphs if not any(kw in p.lower() for kw in self.stop_words)]

        # Добавляем логи для отладки
        print(f"Исходное количество абзацев: {len(input_text.split('\n'))}")
        print(f"Абзацы после фильтрации по минимальной длине: {len(paragraphs)}")
        print(f"Абзацы после фильтрации по ключевым словам: {len(paragraphs)}")

        if not paragraphs:
            return []  # Возвращаем пустой список, если нет подходящих абзацев

        preprocessed_paragraphs = [self.preprocess_text(p) for p in paragraphs]
        paragraph_vectors = self.pipeline.named_steps['tfidf'].transform(preprocessed_paragraphs)
        probabilities = self.pipeline.predict_proba(paragraph_vectors)[:, 1]
        results = [(para, prob) for para, prob in zip(paragraphs, probabilities) if prob >= threshold]

        # Добавляем логи для отладки
        print(f"Абзацы после предобработки: {len(preprocessed_paragraphs)}")
        print(f"Абзацы с вероятностью >= {threshold}: {len(results)}")

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

    def retrain(self, additional_texts, additional_labels):
        self.load_model()
        texts, labels = self.load_data()
        texts.extend(additional_texts)
        labels.extend(additional_labels)
        self.pipeline.fit(texts, labels)
        self.save_model()