import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Загружаем модель и тестовые данные
model = joblib.load('models/model.pkl')
X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv')

# Предсказания
y_pred = model.predict(X_test)

# Вычисляем метрики
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred, average='weighted'),
    "precision": precision_score(y_test, y_pred, average='weighted'),
    "recall": recall_score(y_test, y_pred, average='weighted')
}

# Генерация имени папки для эксперимента
experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_dir = f"experiments/{experiment_name}"
os.makedirs(experiment_dir, exist_ok=True)

# Сохраняем метрики в файл
with open(f'{experiment_dir}/metrics.json', 'w') as f:
    json.dump(metrics, f)

# Генерация графиков
# 1. Матрица ошибок (Confusion Matrix)
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig(f'{experiment_dir}/confusion_matrix.png')
plt.close()

# 2. График распределения вероятностей (Probability Distribution)
probs = model.predict_proba(X_test)
for i in range(probs.shape[1]):
    plt.hist(probs[:, i], bins=20, alpha=0.5, label=f'Class {i}')
plt.title('Probability Distribution')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.legend()
plt.savefig(f'{experiment_dir}/probability_distribution.png')
plt.close()

# 3. Точность по классам
class_precisions = precision_score(y_test, y_pred, average=None)
plt.bar(range(len(class_precisions)), class_precisions)
plt.title('Class-wise Precision')
plt.xlabel('Class')
plt.ylabel('Precision')
plt.savefig(f'{experiment_dir}/class_precision.png')
plt.close()

print(f"Experiment saved in: {experiment_dir}")
