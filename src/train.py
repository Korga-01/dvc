import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import yaml

# Загрузка гиперпараметров
params = yaml.safe_load(open('params.yaml'))['train']

# Загрузка данных
X_train = pd.read_csv('data/processed/X_train.csv')
y_train = pd.read_csv('data/processed/y_train.csv')

# Обучение модели
model = RandomForestClassifier(
    n_estimators=params['n_estimators'],
    max_depth=params['max_depth'],
    random_state=params['random_state']
)
model.fit(X_train, y_train.values.ravel())

# Сохранение модели
joblib.dump(model, 'models/model.pkl')
