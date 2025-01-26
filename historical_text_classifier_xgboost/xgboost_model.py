import json
import numpy as np
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from .utils import preprocess_text, extract_valuable_passages


class XGBoostTextClassifier:
    def __init__(self, model_path='historical_text_classifier_xgboost/model.joblib',
                 dataset_path='historical_text_classifier_xgboost/data/dataset.json'):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.pipeline = None
        self.is_trained = False
        self.load_data()
        self.create_pipeline()

    def load_data(self):
        """Загрузка данных из файла."""
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.texts = [item["text"] for item in data]
        self.labels = [1 if item["label"] == "historical_background" else 0 for item in data]

    def create_pipeline(self):
        """Создание конвейера для обработки текста и классификации."""
        self.pipeline = Pipeline(steps=[
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('classifier', XGBClassifier(random_state=42))
        ])

    def train(self):
        """Обучение модели на данных."""
        X_train, X_test, y_train, y_test = train_test_split(self.texts, self.labels, test_size=0.2, random_state=42)
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.01, 0.1],
            'classifier__max_depth': [3, 6, 9]
        }

        grid_search = GridSearchCV(self.pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        self.pipeline = grid_search.best_estimator_
        self.is_trained = True
        self.save_model()

    def save_model(self):
        """Сохранение обученной модели."""
        joblib.dump(self.pipeline, self.model_path)

    def load_model(self):
        """Загрузка модели, если она существует."""
        if os.path.exists(self.model_path):
            self.pipeline = joblib.load(self.model_path)
            self.is_trained = True
        else:
            self.train()

    def predict(self, text):
        """Предсказание класса для нового текста."""
        if not self.is_trained:
            raise Exception("Модель еще не обучена.")
        return self.pipeline.predict([text])[0]

    def extract_valuable_passages(self, text, threshold=0.5):
        """Извлечение значимых отрывков из текста."""
        return extract_valuable_passages(text, self.pipeline, threshold)

    def update_model(self, new_data):
        """Дополнительное обучение модели на новых данных."""
        new_texts = [item["text"] for item in new_data]
        new_labels = [1 if item["label"] == "historical_background" else 0 for item in new_data]

        self.texts.extend(new_texts)
        self.labels.extend(new_labels)

        # Повторное обучение модели
        self.train()

    def retrain(self):
        """Перемешивание и повторное обучение модели с использованием прежних данных."""
        self.train()

    def plot_feature_importance(self):
        """Построение графика важности признаков."""
        if not self.is_trained:
            raise Exception("Модель еще не обучена.")

        booster = self.pipeline.named_steps['classifier'].get_booster()
        importance = booster.get_score(importance_type='weight')
        importance = dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))

        feature_names = self.pipeline.named_steps['tfidf'].get_feature_names_out()
        feature_importances = np.array([importance.get(f, 0) for f in feature_names])

        # Выбор топ-30 признаков
        top_n = 30
        sorted_indices = feature_importances.argsort()[::-1][:top_n]
        sorted_importances = feature_importances[sorted_indices]
        sorted_feature_names = feature_names[sorted_indices]

        # Построение графика
        plt.figure(figsize=(12, 6))
        plt.bar(range(top_n), sorted_importances, align="center")
        plt.xticks(range(top_n), sorted_feature_names, rotation=90)
        plt.xlabel("Слова")
        plt.ylabel("Важность")
        plt.title("Важность признаков (слова)")
        plt.show()