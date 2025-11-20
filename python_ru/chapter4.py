# -*- coding: utf-8 -*-
"""
Код к Главе 4
====================

Анализ и визуализация данных с использованием Python.
"""
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import OLSInfluence
from pygam import LinearGAM, s, f

import seaborn as sns
import matplotlib.pylab as plt
from tabulate import tabulate

# Определение путей к данным и рисункам
PSDS_PATH = os.path.join(os.path.expanduser("~"), "statistics-for-data-scientists")
FIGURES_PATH = os.path.join(PSDS_PATH, "figures")
DATA_PATH = os.path.join(PSDS_PATH, "data")


# Загрузка данных
lung = pd.read_csv(os.path.join(DATA_PATH, "LungDisease.csv"))
house = pd.read_csv(os.path.join(DATA_PATH, "house_sales.csv"), sep="\t")

# ====== Простая линейная регрессия ======

# Рисунок 4.1: Диаграмма рассеяния для данных о заболевании легких
lung.plot.scatter(x="Exposure", y="PEFR", figsize=(5, 5))
plt.xlabel("Воздействие")
plt.ylabel("ПСВ (Пиковая скорость выдоха)")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, "psds_0401.png"))
plt.close()

# Модель линейной регрессии
model = smf.ols("PEFR ~ Exposure", data=lung).fit()
print("Параметры модели линейной регрессии:")
print(model.summary())

# Рисунок 4.2: Линия регрессии
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(0, 23)
ax.set_ylim(250, 500)
ax.plot(lung.Exposure, lung.PEFR, "o")
ax.plot(lung.Exposure, model.predict(lung.Exposure), color="blue")
ax.set_xlabel("Воздействие")
ax.set_ylabel("ПСВ")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, "psds_0402.png"))
plt.close()


# ====== Множественная линейная регрессия ======

predictors = ["SqFtTotLiving", "SqFtLot", "Bathrooms", "Bedrooms", "BldgGrade"]
outcome = "AdjSalePrice"

house_lm = smf.ols(f"{outcome} ~ {' + '.join(predictors)}", data=house).fit()
print("\nМодель множественной регрессии для цен на дома:")
print(house_lm.summary())


# ====== Пошаговая регрессия ======

def stepwise_selection(X, y, initial_list=[], threshold_in=0.01, threshold_out=0.05, verbose=True):
    included = list(initial_list)
    while True:
        changed = False
        # Прямой шаг
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded, dtype=np.float64)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print(f'Добавлен {best_feature} с p-value {best_pval:.6f}')

        # Обратный шаг
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()
        if worst_pval > threshold_out:
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            changed = True
            if verbose:
                print(f'Удален {worst_feature} с p-value {worst_pval:.6f}')
        if not changed:
            break
    return included

print("\nПроведение пошаговой регрессии...")
# Подготовка данных для пошаговой регрессии
predictors_full = ['SqFtTotLiving', 'SqFtLot', 'Bathrooms', 'Bedrooms', 'BldgGrade', 'PropertyType', 'NbrLivingUnits',
                 'SqFtFinBasement', 'YrBuilt', 'YrRenovated', 'NewConstruction']
X = house[predictors_full]
y = house[outcome]
# Обработка категориальных переменных и пропусков
X = pd.get_dummies(X, columns=['PropertyType'], drop_first=True)
X['NewConstruction'] = X['NewConstruction'].astype(int)
X = X.fillna(0)

final_predictors = stepwise_selection(X, y)
print("\nИтоговые предикторы после пошагового отбора:")
print(final_predictors)

house_step_lm = smf.ols(f'{outcome} ~ {" + ".join(final_predictors)}', data=pd.concat([X, y], axis=1)).fit()
print("\nМодель после пошаговй регрессии:")
print(house_step_lm.summary())


# ====== Диагностика регрессии ======

house_98105 = house.loc[house['ZipCode'] == 98105, ]
predictors_98105 = ['SqFtTotLiving', 'SqFtLot', 'Bathrooms', 'Bedrooms', 'BldgGrade']

lm_98105 = smf.ols(f'AdjSalePrice ~ {" + ".join(predictors_98105)}', data=house_98105).fit()

influence = OLSInfluence(lm_98105)
sresiduals = influence.resid_studentized_internal

# Рисунок 4.5: Влиятельная точка
np.random.seed(11)
x = np.random.normal(size=25)
y = -x / 5 + np.random.normal(size=25)
x[0] = 8
y[0] = 8

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(x, y, 'o')

model1 = LinearRegression()
model1.fit(x.reshape(-1, 1), y)
ax.plot(x, model1.predict(x.reshape(-1, 1)), color='blue')

model2 = LinearRegression()
model2.fit(x[1:].reshape(-1, 1), y[1:])
ax.plot(x[1:], model2.predict(x[1:].reshape(-1, 1)), color='red', linestyle='--')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, "psds_0405.png"))
plt.close()

# Рисунок 4.6: График влиятельности
hat_values = influence.hat_matrix_diag
cooks_d = influence.cooks_distance[0]

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(hat_values, sresiduals, s=cooks_d*1000)
ax.axhline(y=-2.5, linestyle='--', color='darkgrey')
ax.axhline(y=2.5, linestyle='--', color='darkgrey')
ax.set_xlabel("Leverage (Рычаг)")
ax.set_ylabel("Standardized Residuals (Стандартизированные остатки)")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, "psds_0406.png"))
plt.close()

# Рисунок 4.7: Гетероскедастичность
fig, ax = plt.subplots(figsize=(5, 5))
sns.regplot(x=lm_98105.fittedvalues, y=np.abs(lm_98105.resid), scatter_kws={'alpha': 0.25}, line_kws={'color': 'C1'}, lowess=True, ax=ax)
ax.set_xlabel("Predicted values (Предсказанные значения)")
ax.set_ylabel("Absolute Residuals (Абсолютные остатки)")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, "psds_0407.png"))
plt.close()

# Рисунок 4.9: График частичных остатков
fig = sm.graphics.plot_partregress_grid(lm_98105, fig=plt.figure(figsize=(10, 8)))
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, "psds_0409.png"))
plt.close()


# ====== Полиномиальная и сплайновая регрессия ======

# Полиномиальная регрессия
formula_poly = 'AdjSalePrice ~ SqFtTotLiving + SqFtLot + Bathrooms + Bedrooms + BldgGrade + np.power(SqFtTotLiving, 2)'
lm_poly = smf.ols(formula=formula_poly, data=house_98105).fit()

print("\nПолиномиальная регрессия:")
print(lm_poly.summary())

# Рисунок 4.10: График частичных остатков для полиномиальной модели
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot()
sm.graphics.plot_ccpr(lm_poly, "np.power(SqFtTotLiving, 2)", ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, "psds_0410.png"))
plt.close()

# Сплайновая регрессия
from patsy import bs
formula_spline = 'AdjSalePrice ~ bs(SqFtTotLiving, knots=[2000, 2500], degree=3, include_intercept=False) + SqFtLot + Bathrooms + Bedrooms + BldgGrade'
lm_spline = smf.ols(formula=formula_spline, data=house_98105).fit()

print("\nСплайновая регрессия:")
print(lm_spline.summary())

# Рисунок 4.12: График частичных остатков для сплайновой модели
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot()
sm.graphics.plot_ccpr(lm_spline, "bs(SqFtTotLiving, knots=[2000, 2500], degree=3, include_intercept=False)", ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, "psds_0412.png"))
plt.close()


# ====== Обобщенные аддитивные модели (GAM) ======

predictors_gam = ['SqFtLot', 'Bathrooms', 'Bedrooms', 'BldgGrade']
X_gam = house_98105[predictors_gam].values
y_gam = house_98105['AdjSalePrice'].values

X_s_gam = house_98105['SqFtTotLiving'].values

gam = LinearGAM(s(0, n_splines=12) + f(1) + f(2) + f(3) + f(4)).fit(np.hstack([X_s_gam[:, np.newaxis], X_gam]), y_gam)

print("\nОбобщенная аддитивная модель (GAM):")
print(gam.summary())

# Рисунок 4.13: График для GAM
fig, axes = plt.subplots(1, 1, figsize=(6, 6))
titles = ['SqFtTotLiving']

for i, ax in enumerate([axes]):
    XX = gam.generate_X_grid(term=i)
    pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
    
    ax.plot(XX[:, i], pdep)
    ax.plot(XX[:, i], confi, c='r', ls='--')
    
    # Добавим точки остатков (rug plot)
    sns.rugplot(x=X_s_gam, ax=ax, color='k', alpha=0.3)
    
    ax.set_title(titles[i])
    
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, "psds_0413.png"))
plt.close()

