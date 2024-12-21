import pandas as pd
from sklearn.model_selection import train_test_split

params = {
    "test_size": 0.2,
    "random_state": 42
}

# Загрузка данных
data = pd.read_csv('data/raw/iris.csv')

# Разделение на признаки и метки
X = data.drop('species', axis=1)
y = data['species']

# Разделение данных на тренировочные и тестовые
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params['test_size'], random_state=params['random_state'])

# Сохранение подготовленных данных
X_train.to_csv('data/processed/X_train.csv', index=False)
X_test.to_csv('data/processed/X_test.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)
