# -*- coding: utf-8 -*-
"""
Код к Главе 3
====================

Анализ и визуализация данных с использованием Python.
"""
import os
import random
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pylab as plt

# Определение путей к данным и рисункам
PSDS_PATH = os.path.join(os.path.expanduser("~"), "statistics-for-data-scientists")
FIGURES_PATH = os.path.join(PSDS_PATH, "figures")
DATA_PATH = os.path.join(PSDS_PATH, "data")


# Загрузка данных
session_times = pd.read_csv(os.path.join(DATA_PATH, "web_page_data.csv"))
session_times["Time"] = session_times["Time"] * 100  # Перевод времени в секунды
four_sessions = pd.read_csv(os.path.join(DATA_PATH, "four_sessions.csv"))
click_rate = pd.read_csv(os.path.join(DATA_PATH, "click_rates.csv"))
imanishi_data = pd.read_csv(os.path.join(DATA_PATH, "imanishi_data.csv"))

# ====== Перестановочные тесты ======

# Рисунок 3.3: Время сеанса для страниц A и B
ax = session_times.boxplot(by="Page", column="Time", figsize=(5, 5))
ax.set_xlabel("")
ax.set_ylabel("Время (в секундах)")
plt.suptitle("")
plt.title("")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, "psds_0303.png"))
plt.close()

# Среднее время для каждой страницы
mean_a = session_times[session_times.Page == "Page A"].Time.mean()
mean_b = session_times[session_times.Page == "Page B"].Time.mean()
observed_diff = mean_b - mean_a
print(f"Наблюдаемая разница средних: {observed_diff:.2f}")

# Функция для перестановочного теста
def perm_fun(x, nA, nB):
    n = nA + nB
    idx_b = set(random.sample(range(n), nB))
    idx_a = set(range(n)) - idx_b
    return x.loc[list(idx_b)].mean() - x.loc[list(idx_a)].mean()

# Проведение перестановочного теста для времени сеансов
perm_diffs = [
    perm_fun(session_times.Time, 21, 15) for _ in range(1000)
]

# Рисунок 3.4: Распределение перестановок
fig, ax = plt.subplots(figsize=(5, 5))
ax.hist(perm_diffs, bins=11, rwidth=0.9)
ax.axvline(x=observed_diff, color="black", lw=2)
ax.text(60, 200, "Наблюдаемая\nразница", bbox={"facecolor": "white"})
ax.set_xlabel("Разница во времени сеансов (в секундах)")
ax.set_ylabel("Частота")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, "psds_0304.png"))
plt.close()

# Расчет p-value
p_value = np.mean(np.array(perm_diffs) > observed_diff)
print(f"P-value для перестановочного теста: {p_value}")


# ====== Статистическая значимость и p-value ======

# Тест пропорций для конверсии
print("\nТест пропорций (z-тест):")
survivors = np.array([[200, 23739 - 200], [182, 22588 - 182]])
z_stat, p_value = sm.stats.proportions_ztest(survivors[:, 0], np.sum(survivors, axis=1), alternative='larger')
print(f'Z-статистика: {z_stat:.2f}, p-value: {p_value:.4f}')

# t-тест для времени сеансов
res = stats.ttest_ind(
    session_times[session_times.Page == "Page A"].Time,
    session_times[session_times.Page == "Page B"].Time,
    equal_var=False,
    alternative='less' # H1: mean(A) < mean(B)
)
print(f"\nt-тест: t={res.statistic:.2f}, p-value={res.pvalue:.4f}")


# ====== ANOVA ======

# Рисунок 3.6: Ящик с усами для четырех групп сеансов
four_sessions.boxplot(by="Page", column="Time", figsize=(5, 5))
plt.suptitle("")
plt.title("")
plt.xlabel("")
plt.ylabel("Время (в секундах)")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, "psds_0306.png"))
plt.close()

# ANOVA тест
observed_variance = four_sessions.groupby("Page").mean().var()[0]
print(f"Наблюдаемая дисперсия средних: {observed_variance:.4f}")

def perm_test_anova(df):
    df = df.copy()
    df["Time"] = np.random.permutation(df["Time"].values)
    return df.groupby("Page").mean().var()[0]

perm_variance = [perm_test_anova(four_sessions) for _ in range(3000)]
p_value_anova = np.mean(np.array(perm_variance) > observed_variance)
print(f"P-value (перестановочный тест для ANOVA): {p_value_anova:.4f}")

# Классический F-тест
f_stat, p_value_f = stats.f_oneway(
    four_sessions[four_sessions.Page == "Page 1"].Time,
    four_sessions[four_sessions.Page == "Page 2"].Time,
    four_sessions[four_sessions.Page == "Page 3"].Time,
    four_sessions[four_sessions.Page == "Page 4"].Time
)
print(f"F-статистика: {f_stat:.2f}, p-value: {p_value_f:.4f}")


# ====== Критерий хи-квадрат ======

# Создание таблицы сопряженности
clicks = click_rate.pivot(index="Headline", columns="Click", values="Rate")
print("\nТаблица сопряженности для кликов:")
print(clicks)

# Хи-квадрат тест
chi2, p_value, _, _ = stats.chi2_contingency(clicks)
print(f"\nХи-квадрат: {chi2:.2f}, p-value: {p_value:.4f}")

# Рисунок 3.7: Распределения хи-квадрат
x = np.linspace(1, 30, 100)
dfs = [1, 2, 5, 20]

plt.figure(figsize=(6, 4))
for df in dfs:
    plt.plot(x, stats.chi2.pdf(x, df), label=f"df={df}")

plt.xlabel("Значение")
plt.ylabel("Плотность вероятности")
plt.legend()
plt.savefig(os.path.join(FIGURES_PATH, "psds_0307.png"))
plt.close()


# ====== Точный тест Фишера ======

odds_ratio, p_value = stats.fisher_exact(clicks)
print(f"\nТочный тест Фишера: p-value = {p_value:.4f}")


# ====== Пример с данными Tufts ======

# Рисунок 3.8: Распределение первых цифр
plt.figure(figsize=(5, 5))
imanishi_data.plot.bar(x="Digit", y="Frequency", legend=False)
plt.xlabel("Цифра")
plt.ylabel("Частота")

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, "psds_0308.png"))
plt.close()

