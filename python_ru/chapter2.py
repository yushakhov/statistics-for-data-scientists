# -*- coding: utf-8 -*-
"""
Код к Главе 2
====================

Анализ и визуализация данных с использованием Python.
"""
import os
import pandas as pd
import numpy as np
from sklearn.utils import resample
from scipy import stats
import seaborn as sns
import matplotlib.pylab as plt

# Определение путей к данным и рисункам
PSDS_PATH = os.path.join(os.path.expanduser("~"), "statistics-for-data-scientists")
FIGURES_PATH = os.path.join(PSDS_PATH, "figures")
DATA_PATH = os.path.join(PSDS_PATH, "data")

# Загрузка данных
loans_income = pd.read_csv(os.path.join(DATA_PATH, "loans_income.csv")).squeeze("columns")
sp500_px = pd.read_csv(os.path.join(DATA_PATH, "sp500_px.csv"), index_col=0)


# ====== Bootstrap ======

# Создание 1000 bootstrap-выборок медиан
results = []
for n_repeat in range(1000):
    sample = resample(loans_income)
    results.append(sample.median())
results = pd.Series(results)

print("Статистика по bootstrap-выборкам:")
print(f"Исходная медиана: {loans_income.median()}")
print(f"Смещение (Bias): {results.mean() - loans_income.median()}")
print(f"Стандартная ошибка: {results.std()}")
print("\n")


# ====== Центральная предельная теорема ======

# Создание выборок для демонстрации ЦПТ
sample_data = pd.DataFrame({
    'income': loans_income.sample(1000),
    'type': 'Data',
})

sample_mean_05 = pd.DataFrame({
    'income': [loans_income.sample(5).mean() for _ in range(1000)],
    'type': 'Mean of 5',
})

sample_mean_20 = pd.DataFrame({
    'income': [loans_income.sample(20).mean() for _ in range(1000)],
    'type': 'Mean of 20',
})

results = pd.concat([sample_data, sample_mean_05, sample_mean_20])

# Рисунок 2.6: Гистограммы для демонстрации ЦПТ
g = sns.FacetGrid(results, col="type", col_wrap=1, height=3, aspect=2)
g.map(plt.hist, "income", range=[0, 200000], bins=40)
g.set_axis_labels("Доход", "Количество")
g.set_titles("{col_name}")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, "psds_0206.png"))
plt.close()


# ====== Нормальное распределение ======

# Рисунок: Плотность нормального распределения
fig, ax = plt.subplots(figsize=(4, 4))
x = np.linspace(-3, 3, 300)
pdf = stats.norm.pdf(x)
ax.plot(x, pdf)
ax.fill_between(x, pdf, color='blue', alpha=0.5)
ax.set_axis_off()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'normal_density.png'))
plt.close()


# ====== Квантиль-квантильный график (Q-Q plot) ======

# Рисунок 2.11: Q-Q plot для нормально распределенных данных
fig, ax = plt.subplots(figsize=(4, 4))
norm_sample = stats.norm.rvs(size=100)
stats.probplot(norm_sample, plot=ax)
ax.set_title("")
ax.set_xlabel("")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'psds_0211.png'))
plt.close()

# Рисунок 2.12: Q-Q plot для доходности акций NFLX
nflx = sp500_px.NFLX
nflx = np.diff(np.log(nflx[nflx > 0]))

fig, ax = plt.subplots(figsize=(4, 4))
stats.probplot(nflx, plot=ax)
ax.set_title("")
ax.set_xlabel("")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'psds_0212.png'))
plt.close()

