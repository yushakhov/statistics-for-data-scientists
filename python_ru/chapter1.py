# -*- coding: utf-8 -*-
"""
Код к Главе 1
====================

Наука о данных для чайников, Первое издание

Анализ и визуализация данных с использованием Python.

Материалы к книге:
https://www.oreilly.com/library/view/practical-statistics-for/9781491952955/

Оригинальный код на R:
https://github.com/gedeck/practical-statistics-for-data-scientists

Это Python-адаптация оригинального кода на R.
"""
import os
import pandas as pd
import numpy as np
from scipy.stats import trim_mean
from statsmodels import robust
import seaborn as sns
import matplotlib.pylab as plt
from tabulate import tabulate

# Определение путей к данным и рисункам
PSDS_PATH = os.path.join(os.path.expanduser("~"), "statistics-for-data-scientists")
FIGURES_PATH = os.path.join(PSDS_PATH, "figures")
DATA_PATH = os.path.join(PSDS_PATH, "data")

# Создание директории для рисунков, если она не существует
os.makedirs(FIGURES_PATH, exist_ok=True)

# Загрузка данных
state = pd.read_csv(os.path.join(DATA_PATH, "state.csv"))
dfw_airline = pd.read_csv(os.path.join(DATA_PATH, "dfw_airline.csv"), index_col=0)
sp500_px = pd.read_csv(os.path.join(DATA_PATH, "sp500_px.csv"), index_col=0)
sp500_sym = pd.read_csv(os.path.join(DATA_PATH, "sp500_sym.csv"))
kc_tax = pd.read_csv(os.path.join(DATA_PATH, "kc_tax.csv"))
lc_loans = pd.read_csv(os.path.join(DATA_PATH, "lc_loans.csv"))
airline_stats = pd.read_csv(os.path.join(DATA_PATH, "airline_stats.csv"))

# Преобразование столбца "airline" в категориальный с определенным порядком
airline_stats["airline"] = pd.Categorical(
    airline_stats["airline"],
    categories=["Alaska", "American", "Jet Blue", "Delta", "United", "Southwest"],
    ordered=True,
)

# Таблица: Несколько строк из датафрейма state
print("Таблица: Несколько строк из датафрейма state")
state_head = state.head(8).copy()
state_head["Population"] = state_head["Population"].apply(lambda x: f"{x:,}")
print(
    tabulate(
        state_head, headers="keys", tablefmt="pipe", stralign="right", numalign="right"
    )
)
print("\n")


# ====== Оценки центрального положения ======

print("Среднее значение численности населения:", state["Population"].mean())
print("Усеченное среднее (10%):", trim_mean(state["Population"], 0.1))
print("Медиана численности населения:", state["Population"].median())
print("\n")

print("Средний уровень убийств:", state["Murder.Rate"].mean())
print(
    "Средневзвешенный уровень убийств:",
    np.average(state["Murder.Rate"], weights=state["Population"]),
)
print("\n")


# ====== Оценки вариации ======

print("Стандартное отклонение численности населения:", state["Population"].std())
print("Межквартильный размах (IQR):", state["Population"].quantile(0.75) - state["Population"].quantile(0.25))
print("Медианное абсолютное отклонение (MAD):", robust.mad(state["Population"]))
print("\n")


# ====== Распределение данных ======

# Таблица: Процентили уровня убийств по штатам
percentiles = state["Murder.Rate"].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
print("Процентили уровня убийств:")
print(tabulate(pd.DataFrame(percentiles), headers="keys", tablefmt="pipe"))
print("\n")

# Рисунок 1.2: Ящик с усами для численности населения
plt.figure(figsize=(4, 5))
ax = state["Population"].plot.box()
ax.set_ylabel("Население (в миллионах)")
ax.set_yticklabels([f"{int(y/1_000_000)}" for y in ax.get_yticks()])
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, "psds_0102.png"))
plt.close()

# Таблица: Таблица частот для численности населения
binned_population = pd.cut(state["Population"], 10)
freq_table = binned_population.value_counts().reset_index()
freq_table.columns = ["Диапазон бинов", "Частота"]
freq_table = freq_table.sort_values(by="Диапазон бинов")
print("Таблица частот для численности населения:")
print(tabulate(freq_table, headers="keys", tablefmt="pipe", showindex=False))
print("\n")

# Рисунок 1.3: Гистограмма численности населения
plt.figure(figsize=(5, 4))
ax = state["Population"].plot.hist(figsize=(5, 4))
ax.set_xlabel("Население (в миллионах)")
ax.set_xticklabels([f"{int(x/1_000_000)}" for x in ax.get_xticks()])
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, "psds_0103.png"))
plt.close()

# Рисунок 1.4: Гистограмма и оценка плотности для уровня убийств
plt.figure(figsize=(5, 4))
ax = state["Murder.Rate"].plot.hist(density=True, xlim=[0, 12], bins=range(1, 12))
state["Murder.Rate"].plot.density(ax=ax)
ax.set_xlabel("Уровень убийств (на 100,000)")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, "psds_0104.png"))
plt.close()


# ====== Изучение бинарных и категориальных данных ======

print("Причины задержек рейсов в аэропорту DFW:")
print(dfw_airline / dfw_airline.to_numpy().sum() * 100)
print("\n")

# Рисунок 1.5: Столбчатая диаграмма причин задержек
plt.figure(figsize=(5, 4))
ax = (dfw_airline.sum() / 6).plot.bar(figsize=(5, 4), rot=0)
ax.set_ylabel("Количество задержек")
ax.tick_params(axis="x", labelsize=8)
ax.tick_params(axis="y", labelsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, "psds_0105.png"))
plt.close()


# ====== Корреляция ======

# Выбор акций телекоммуникационного сектора
telecom_symbols = sp500_sym[sp500_sym["sector"] == "telecommunications_services"]["symbol"]
telecom = sp500_px.loc[sp500_px.index >= "2012-07-01", telecom_symbols]

print("Корреляционная матрица для акций телекоммуникационного сектора:")
print(telecom.corr())
print("\n")

# Выбор акций ETF
etfs = sp500_px.loc[sp500_px.index > "2012-07-01", sp500_sym[sp500_sym["sector"] == "etf"]["symbol"]]

# Рисунок 1.6: Тепловая карта корреляции для ETF
fig, ax = plt.subplots(figsize=(5, 4))
ax = sns.heatmap(etfs.corr(), vmin=-1, vmax=1, cmap="viridis", annot=True, ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, "psds_0106.png"))
plt.close()


# ====== Изучение двух или более переменных ======

# Рисунок 1.7: Диаграмма рассеяния для акций T и VZ
ax = telecom.plot.scatter(x="T", y="VZ", figsize=(5, 4), marker="$\u25EF$")
ax.set_xlabel("AT&T (T)")
ax.set_ylabel("Verizon (VZ)")
ax.axhline(0, color="grey", lw=1)
ax.axvline(0, color="grey", lw=1)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, "psds_0107.png"))
plt.close()

# Фильтрация данных по налогу на недвижимость в округе Кинг
kc_tax0 = kc_tax.loc[
    (kc_tax.TaxAssessedValue < 750000)
    & (kc_tax.SqFtTotLiving > 100)
    & (kc_tax.SqFtTotLiving < 3500),
    :,
]
print("Количество записей после фильтрации:", len(kc_tax0))

# Рисунок 1.8: Гексагональная диаграмма
fig, ax = plt.subplots(figsize=(5, 4))
hb = ax.hexbin(
    kc_tax0.SqFtTotLiving,
    kc_tax0.TaxAssessedValue,
    gridsize=30,
    cmap="viridis",
    mincnt=1,
)
ax.set_xlabel("Жилая площадь (кв. футы)")
ax.set_ylabel("Оценочная стоимость (долл. США)")
cb = fig.colorbar(hb, ax=ax)
cb.set_label("count")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, "psds_0108.png"))
plt.close()

# Рисунок 1.9: Контурный график
fig, ax = plt.subplots(figsize=(5, 4))
sns.kdeplot(
    data=kc_tax0,
    x="SqFtTotLiving",
    y="TaxAssessedValue",
    ax=ax,
    cmap="viridis",
    fill=True,
)
ax.set_xlabel("Жилая площадь (кв. футы)")
ax.set_ylabel("Оценочная стоимость (долл. США)")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, "psds_0109.png"))
plt.close()

# Таблица сопряженности
crosstab = lc_loans.pivot_table(index="grade", columns="status", aggfunc=lambda x: len(x), margins=True)
print("Таблица сопряженности для кредитов:")
print(crosstab)
print("\n")

df = crosstab.loc["A":"G", :].copy()
df.loc[:, "Charged Off":"Late"] = df.loc[:, "Charged Off":"Late"].div(df["All"], axis=0)
df["All"] = df["All"] / sum(df["All"])
perc_crosstab = df
print("Таблица сопряженности (в процентах):")
print(perc_crosstab.round(4) * 100)
print("\n")


# ====== Категориальные и числовые данные ======

# Рисунок 1.10: Ящик с усами для процента задержек по вине перевозчика
plt.figure(figsize=(5, 4))
ax = sns.boxplot(x="airline", y="pct_carrier_delay", data=airline_stats, color="lightgrey")
ax.set_xlabel("")
ax.set_ylabel("Ежедневный % задержанных рейсов")
plt.ylim([0, 50])
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, "psds_0110.png"))
plt.close()

# Рисунок 1.11: Скрипичный график
plt.figure(figsize=(5, 4))
ax = sns.violinplot(
    x="airline", y="pct_carrier_delay", data=airline_stats, inner="quartile", color="white"
)
ax.set_xlabel("")
ax.set_ylabel("Ежедневный % задержанных рейсов")
plt.ylim([0, 50])
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, "psds_0111.png"))
plt.close()


# ====== Визуализация нескольких переменных ======

# Рисунок 1.12: Фасетный график для данных о налогах
zip_codes = [98188, 98105, 98108, 98126]
kc_tax_zip = kc_tax0.loc[kc_tax0.ZipCode.isin(zip_codes), :]

def hexbin(x, y, color, **kwargs):
    cmap = sns.light_palette(color, as_cmap=True)
    plt.hexbin(x, y, gridsize=25, cmap=cmap, **kwargs)

g = sns.FacetGrid(kc_tax_zip, col="ZipCode", col_wrap=2)
g.map(hexbin, "SqFtTotLiving", "TaxAssessedValue", mincnt=1, extent=[0, 3500, 0, 700000])
g.set_axis_labels("Жилая площадь (кв. футы)", "Оценочная стоимость (долл. США)")
g.set_titles("ZipCode {col_name}")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, "psds_0112.png"))
plt.close()

