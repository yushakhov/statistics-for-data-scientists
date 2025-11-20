# -*- coding: utf-8 -*-
"""
Код к Главе 6
====================

Анализ и визуализация данных с использованием Python.
"""
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

import seaborn as sns
import matplotlib.pylab as plt

# Определение путей к данным и рисункам
PSDS_PATH = os.path.join(os.path.expanduser("~"), "statistics-for-data-scientists")
FIGURES_PATH = os.path.join(PSDS_PATH, "figures")
DATA_PATH = os.path.join(PSDS_PATH, "data")


# Загрузка данных
loan200 = pd.read_csv(os.path.join(DATA_PATH, "loan200.csv"))
loan3000 = pd.read_csv(os.path.join(DATA_PATH, "loan3000.csv"))

loan_data = pd.read_csv(os.path.join(DATA_PATH, "loan_data.csv"))
loan_data = loan_data.drop(columns=["Unnamed: 0", "status"])
loan_data["outcome"] = pd.Categorical(loan_data["outcome"], categories=["paid off", "default"], ordered=True)


# ====== K-ближайших соседей (KNN) ======

predictors = ['payment_inc_ratio', 'dti']
outcome = 'outcome'

newloan = loan200.loc[0:0, predictors]
X = loan200.loc[1:, predictors]
y = loan200.loc[1:, outcome]

knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X, y)

distances, indices = knn.kneighbors(newloan)

print("Предсказание для нового кредита (KNN):", knn.predict(newloan)[0])

# Рисунок 6.2: Пример KNN
fig, ax = plt.subplots(figsize=(6, 5))
sns.scatterplot(x='payment_inc_ratio', y='dti', style='outcome', hue='outcome', 
                data=loan200, ax=ax, s=70)
ax.scatter(newloan.payment_inc_ratio, newloan.dti, marker='x', color='red', s=100)

# Окружность вокруг ближайших соседей
max_dist = np.max(distances)
circle = plt.Circle((newloan.payment_inc_ratio.iloc[0], newloan.dti.iloc[0]), max_dist, 
                  color='black', fill=False)
ax.add_artist(circle)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'psds_0602.png'))
plt.close()

# Стандартизация (масштабирование)
loan_df = loan_data[predictors]
loan_outcome = loan_data[outcome]

scaler = StandardScaler()
loan_scaled = scaler.fit_transform(loan_df)

knn_std = KNeighborsClassifier(n_neighbors=5)
knn_std.fit(loan_scaled, loan_outcome)

print("\nПример предсказания на стандартизированных данных:",
      knn_std.predict(scaler.transform(newloan))[0])


# ====== Деревья решений ======

X_tree = loan3000[['borrower_score', 'payment_inc_ratio']]
y_tree = (loan3000.outcome == 'default').astype(int)

loan_tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, min_impurity_decrease=0.005)
loan_tree.fit(X_tree, y_tree)

# Рисунок 6.3: Визуализация дерева решений
plt.figure(figsize=(12, 8))
plot_tree(loan_tree, feature_names=X_tree.columns, class_names=['paid off', 'default'],
          filled=True, rounded=True, label='all', impurity=False, proportion=True)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'psds_rpart_tree.png'))
plt.close()

# Рисунок 6.5: Сравнение мер неопределенности
def gini_impurity(p):
    return p * (1 - p)

def entropy(p):
    if p == 0 or p == 1:
        return 0
    return -p * np.log2(p) - (1-p) * np.log2(1-p)

p_values = np.linspace(0, 1, 101)
gini_values = [gini_impurity(p) for p in p_values]
entropy_values = [entropy(p) for p in p_values]
misclassification_error = [1 - max(p, 1-p) for p in p_values]

plt.figure(figsize=(6, 5))
plt.plot(p_values, gini_values, label='Gini')
plt.plot(p_values, entropy_values, label='Энтропия (Entropy)')
plt.plot(p_values, misclassification_error, label='Ошибка классификации')
plt.xlabel("Доля класса 1 (p)")
plt.ylabel("Мера неопределенности")
plt.legend()
plt.savefig(os.path.join(FIGURES_PATH, 'psds_0605.png'))
plt.close()

# ====== Случайный лес ======

rf = RandomForestClassifier(n_estimators=500, random_state=1)
rf.fit(X_tree, y_tree)

# Рисунок 6.7: Предсказания случайного леса
y_pred_rf = rf.predict(X_tree)

plt.figure(figsize=(6, 5))
sns.scatterplot(x='borrower_score', y='payment_inc_ratio', hue=y_pred_rf, 
                style=y_pred_rf, data=loan3000, alpha=0.5)
plt.xlabel('Borrower Score')
plt.ylabel('Payment-to-Income Ratio')
plt.savefig(os.path.join(FIGURES_PATH, 'psds_0607.png'))
plt.close()

# Важность признаков
predictors_all = ['loan_amnt', 'grade', 'emp_length', 'dti', 'payment_inc_ratio', 
                  'revol_bal', 'revol_util', 'purpose_', 'home_']
X_all = pd.get_dummies(loan_data[predictors_all], drop_first=True)
y_all = (loan_data.outcome == 'default').astype(int)

rf_all = RandomForestClassifier(n_estimators=500, random_state=1)
rf_all.fit(X_all, y_all)

importances = pd.Series(rf_all.feature_importances_, index=X_all.columns).sort_values()

# Рисунок 6.8: Важность признаков
plt.figure(figsize=(6, 10))
importances.plot(kind='barh')
plt.xlabel('Важность (Gini)')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'psds_0608.png'))
plt.close()


# ====== Градиентный бустинг (XGBoost) ======

xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                            subsample=0.63, eta=0.1, n_estimators=100, use_label_encoder=False)
xgb_model.fit(X_tree, y_tree)

# Рисунок 6.9: Предсказания XGBoost
y_pred_xgb = xgb_model.predict(X_tree)

plt.figure(figsize=(6, 5))
sns.scatterplot(x='borrower_score', y='payment_inc_ratio', hue=y_pred_xgb, 
                style=y_pred_xgb, data=loan3000, alpha=0.5)
plt.xlabel('Borrower Score')
plt.ylabel('Payment-to-Income Ratio')
plt.savefig(os.path.join(FIGURES_PATH, 'psds_0609.png'))
plt.close()

# Рисунок 6.10: Кривые обучения
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=10000, random_state=1)

eval_set = [(X_train, y_train), (X_test, y_test)]

# Модель по умолчанию
xgb_default = xgb.XGBClassifier(objective='binary:logistic', eval_metric=['logloss', 'error'], use_label_encoder=False, n_estimators=250)
xgb_default.fit(X_train, y_train, eval_set=eval_set, verbose=False)

# Модель с регуляризацией
xgb_penalty = xgb.XGBClassifier(objective='binary:logistic', eval_metric=['logloss', 'error'], use_label_encoder=False, n_estimators=250,
                              eta=0.1, subsample=0.63, reg_lambda=1000) # lambda is reg_lambda
xgb_penalty.fit(X_train, y_train, eval_set=eval_set, verbose=False)

results_default = xgb_default.evals_result()
results_penalty = xgb_penalty.evals_result()

plt.figure(figsize=(8, 6))
plt.plot(results_default['validation_0']['error'], label='default train')
plt.plot(results_default['validation_1']['error'], label='default test', linestyle='--')
plt.plot(results_penalty['validation_0']['error'], label='penalty train')
plt.plot(results_penalty['validation_1']['error'], label='penalty test', linestyle='--')
plt.xlabel('Итерации')
plt.ylabel('Ошибка')
plt.legend()
plt.savefig(os.path.join(FIGURES_PATH, 'psds_0610.png'))
plt.close()

