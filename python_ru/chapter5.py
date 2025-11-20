# -*- coding: utf-8 -*-
"""
Код к Главе 5
====================

Анализ и визуализация данных с использованием Python.
"""
import os
import pandas as pd
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

import statsmodels.api as sm
import statsmodels.formula.api as smf
from pygam import LogisticGAM, s, f

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

import seaborn as sns
import matplotlib.pylab as plt

# Определение путей к данным и рисункам
PSDS_PATH = os.path.join(os.path.expanduser("~"), "statistics-for-data-scientists")
FIGURES_PATH = os.path.join(PSDS_PATH, "figures")
DATA_PATH = os.path.join(PSDS_PATH, "data")


# Загрузка данных
loan3000 = pd.read_csv(os.path.join(DATA_PATH, "loan3000.csv"))
loan_data = pd.read_csv(os.path.join(DATA_PATH, "loan_data.csv"))
loan_data = loan_data.drop(columns=["Unnamed: 0", "status"])
loan_data["outcome"] = pd.Categorical(loan_data["outcome"], categories=["paid off", "default"], ordered=True)

predictors = ["payment_inc_ratio", "dti", "revol_bal", "revol_util"]
outcome = "outcome"
X = loan_data[predictors]
y = loan_data[outcome]


# ====== Наивный Байес ======

X_nb = loan_data[["purpose_", "home_", "emp_len_"]]
y_nb = loan_data["outcome"]

# Преобразуем категориальные данные в числовые
X_nb = pd.get_dummies(X_nb, drop_first=True)

naive_model = MultinomialNB(alpha=1e-5)
naive_model.fit(X_nb, y_nb)

new_loan = X_nb.iloc[[146]]
print("Пример нового кредита для предсказания:")
print(new_loan)
print("Предсказание Наивного Байеса:", naive_model.predict(new_loan)[0])
print("Вероятности:", naive_model.predict_proba(new_loan))


# ====== Дискриминантный анализ ======
lda = LinearDiscriminantAnalysis()
lda.fit(X, y)

print("\nКоэффициенты ЛДА:")
print(pd.DataFrame(lda.scalings_, index=X.columns))

# Рисунок 5.1: Границы решений для ЛДА
pred = pd.DataFrame(lda.predict_proba(loan3000[predictors]),
                    columns=lda.classes_)

fig, ax = plt.subplots(figsize=(6, 5))
sns.scatterplot(x="borrower_score", y="payment_inc_ratio", 
                hue=pred['default'], data=loan3000, alpha=0.3, ax=ax)
ax.set_xlabel("Borrower Score")
ax.set_ylabel("Payment-to-Income Ratio")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'psds_0501.png'))
plt.close()


# ====== Логистическая регрессия ======

formula = "outcome ~ payment_inc_ratio + purpose_ + home_ + emp_len_ + borrower_score"
model = smf.logit(formula, data=loan_data).fit()
print("\nРезультаты логистической регрессии:")
print(model.summary())

# Рисунок 5.2: Логит-функция
p_values = np.linspace(0.01, 0.99, 100)
logit_values = np.log(p_values / (1 - p_values))
plt.figure(figsize=(6, 5))
plt.plot(p_values, logit_values)
plt.xlabel("p")
plt.ylabel("logit(p)")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'psds_0502.png'))
plt.close()

# Рисунок 5.3: Отношение шансов
odds_values = p_values / (1 - p_values)
plt.figure(figsize=(6, 5))
plt.plot(logit_values, odds_values)
plt.xlabel("log(odds ratio)")
plt.ylabel("odds ratio")
plt.xlim(0, 5)
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'psds_0503.png'))
plt.close()


# ====== GAM для логистической регрессии ======

y_gam = (loan_data.outcome == 'default').astype(int)

X_gam_cat = pd.get_dummies(loan_data[['purpose_', 'home_', 'emp_len_']], drop_first=True)
X_gam_num = loan_data[['payment_inc_ratio', 'borrower_score']]
X_gam = pd.concat([X_gam_num, X_gam_cat], axis=1)

gam = LogisticGAM(s(0) + s(1) + f(2) + f(3) + f(4) + f(5) + f(6) + f(7) + f(8) + f(9)).fit(X_gam, y_gam)

# Рисунок 5.4: График частичной зависимости для GAM
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
titles = ['payment_inc_ratio', 'borrower_score']

for i, ax in enumerate(axes):
    XX = gam.generate_X_grid(term=i)
    pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)

    ax.plot(XX[:, i], pdep)
    ax.plot(XX[:, i], confi, c='r', ls='--')

    sns.rugplot(x=X_gam[X_gam.columns[i]], ax=ax, color='k', alpha=0.3)

    ax.set_title(titles[i])

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, "psds_0504.png"))
plt.close()


# ====== Оценка моделей ======

y_pred = (gam.predict(X_gam) > 0.5).astype(int)

conf_mat = confusion_matrix(y_gam, y_pred)
print("\nМатрица ошибок:")
print(conf_mat)

precision = conf_mat[1, 1] / sum(conf_mat[:, 1])
recall = conf_mat[1, 1] / sum(conf_mat[1, :])
specificity = conf_mat[0, 0] / sum(conf_mat[0, :])

print(f"Точность (Precision): {precision:.4f}")
print(f"Полнота (Recall): {recall:.4f}")
print(f"Специфичность (Specificity): {specificity:.4f}")

# ROC-кривая
y_prob = gam.predict_proba(X_gam)

fpr, tpr, thresholds = roc_curve(y_gam, y_prob)

# Рисунок 5.6: ROC-кривая
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Recall)")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'psds_0506.png'))
plt.close()

auc = roc_auc_score(y_gam, y_prob)
print(f"\nAUC (Площадь под ROC-кривой): {auc:.4f}")

# Рисунок 5.7: Площадь под ROC-кривой
plt.figure(figsize=(6, 5))
plt.fill_between(fpr, tpr, color='blue', alpha=0.3)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Recall)")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'psds_0507.png'))
plt.close()

# ====== Стратегии для несбалансированных данных ======

full_train_set = pd.read_csv(os.path.join(DATA_PATH, "full_train_set.csv"))
y_full = (full_train_set.outcome == 'default').astype(int)

print(f"\nДоля дефолтов в полном наборе данных: {y_full.mean():.4f}")

# Метод взвешивания
weights = np.where(y_full == 1, 1/y_full.mean(), 1)

formula_full = 'outcome ~ payment_inc_ratio + purpose_ + home_ + emp_len_ + dti + revol_bal + revol_util'
full_model_weighted = smf.logit(formula_full, data=full_train_set, freq_weights=weights).fit()

pred_weighted = (full_model_weighted.predict(full_train_set) > 0.5).astype(int)
print(f"Доля предсказанных дефолтов (со взвешиванием): {pred_weighted.mean():.4f}")

# Примечание: для undersampling, oversampling и SMOTE обычно используется библиотека imbalanced-learn.
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.over_sampling import RandomOverSampler, SMOTE
#
# undersample = RandomUnderSampler(sampling_strategy='majority')
# X_under, y_under = undersample.fit_resample(X, y)
#
# oversample = RandomOverSampler(sampling_strategy='minority')
# X_over, y_over = oversample.fit_resample(X, y)
#
# smote = SMOTE(sampling_strategy='minority')
# X_sm, y_sm = smote.fit_resample(X, y)

