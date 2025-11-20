# -*- coding: utf-8 -*-
"""
Код к Главе 7
====================

Анализ и визуализация данных с использованием Python.
"""
import os
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree
from scipy.stats import chi2
import gower

from matplotlib.patches import Ellipse
import seaborn as sns
import matplotlib.pylab as plt

# Определение путей к данным и рисункам
PSDS_PATH = os.path.join(os.path.expanduser("~"), "statistics-for-data-scientists")
FIGURES_PATH = os.path.join(PSDS_PATH, "figures")
DATA_PATH = os.path.join(PSDS_PATH, "data")


# Загрузка данных
sp500_px = pd.read_csv(os.path.join(DATA_PATH, "sp500_px.csv"), index_col=0)
sp500_sym = pd.read_csv(os.path.join(DATA_PATH, "sp500_sym.csv"))


# ====== Метод главных компонент (PCA) ======

oil_px = sp500_px[['XOM', 'CVX']]

pca = PCA(n_components=2)
pca.fit(oil_px)

loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=oil_px.columns)
print("Нагрузки (Loadings) для PCA:")
print(loadings)

# Рисунок 7.1: Главные компоненты для акций нефтяных компаний
def abline(slope, intercept, ax):
    """Рисует линию на графике"""
    x_vals = np.array(ax.get_xlim())
    y_vals = intercept + slope * x_vals
    ax.plot(x_vals, y_vals, '--', color='grey')

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(oil_px['CVX'], oil_px['XOM'], alpha=0.3)
# Добавление эллипса
cov = np.cov(oil_px, rowvar=False)
eigvals, eigvecs = np.linalg.eigh(cov)
order = eigvals.argsort()[::-1]
eigvals, eigvecs = eigvals[order], eigvecs[:, order]

center = oil_px.mean()
angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
chi2_val = chi2.ppf(0.99, df=2)
width, height = 2 * np.sqrt(chi2_val * eigvals)

ellipse = Ellipse(xy=center, width=width, height=height, angle=angle,
                  alpha=0.2, color='grey')
ax.add_patch(ellipse)
# Линии главных компонент
abline(loadings.loc['XOM', 'PC1'] / loadings.loc['CVX', 'PC1'], 0, ax)
abline(loadings.loc['XOM', 'PC2'] / loadings.loc['CVX', 'PC2'], 0, ax)
ax.set_xlabel("CVX")
ax.set_ylabel("XOM")
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, "psds_0701.png"))
plt.close()

# Рисунок 7.2: Scree plot
syms = ['AAPL', 'MSFT', 'CSCO', 'INTC', 'CVX', 'XOM', 'SLB', 'COP',
        'JPM', 'WFC', 'USB', 'AXP', 'WMT', 'TGT', 'HD', 'COST']
top_cons = sp500_px.loc[sp500_px.index >= '2011-01-01', syms]

sp_pca = PCA()
sp_pca.fit(top_cons)

plt.figure(figsize=(5, 4))
plt.plot(range(1, len(sp_pca.explained_variance_ratio_) + 1), 
         sp_pca.explained_variance_ratio_, 'o-')
plt.xlabel('Компонента')
plt.ylabel('Доля объясненной дисперсии')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'psds_0702.png'))
plt.close()

# Рисунок 7.3: Нагрузки компонент
loadings_stocks = pd.DataFrame(sp_pca.components_[:5].T, 
                               columns=[f'PC{i+1}' for i in range(5)],
                               index=top_cons.columns)
loadings_stocks_melted = loadings_stocks.reset_index().melt(id_vars='index')
loadings_stocks_melted.columns = ['Symbol', 'Component', 'Weight']

plt.figure(figsize=(5, 5))
sns.barplot(data=loadings_stocks_melted, x='Symbol', y='Weight', hue='Component')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'psds_0703.png'))
plt.close()


# ====== K-means кластеризация ======

np.random.seed(1010103)
df = sp500_px.loc[sp500_px.index >= '2011-01-01', ['XOM', 'CVX']]

km = KMeans(n_clusters=4, n_init=1, random_state=1010103)
df['cluster'] = km.fit_predict(df)

centers = pd.DataFrame(km.cluster_centers_, columns=['XOM', 'CVX'])
centers['cluster'] = range(4)

# Рисунок 7.4: K-means кластеры
plt.figure(figsize=(5, 4))
sns.scatterplot(data=df, x='XOM', y='CVX', hue='cluster', style='cluster')
sns.scatterplot(data=centers, x='XOM', y='CVX', s=200, marker='X', color='black')
plt.xlim(-2, 2)
plt.ylim(-2.5, 2.5)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'psds_0704.png'))
plt.close()

# Рисунок 7.5: Интерпретация кластеров
syms = ['AAPL', 'MSFT', 'CSCO', 'INTC', 'CVX', 'XOM', 'SLB', 'COP',
        'JPM', 'WFC', 'USB', 'AXP', 'WMT', 'TGT', 'HD', 'COST']
df = sp500_px.loc[sp500_px.index >= '2011-01-01', syms]

np.random.seed(10010)
km = KMeans(n_clusters=5, n_init=10, random_state=10010)
df['cluster'] = km.fit_predict(df)

centers = pd.DataFrame(km.cluster_centers_, columns=syms)
centers_melted = centers.reset_index().melt(id_vars='index')
centers_melted.columns = ['Cluster', 'Symbol', 'Mean']

plt.figure(figsize=(5, 6))
sns.barplot(data=centers_melted, x='Symbol', y='Mean', hue='Cluster')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'psds_0705.png'))
plt.close()

# Рисунок 7.6: Выбор числа кластеров (elbow plot)
pct_var = []
total_ss = KMeans(n_clusters=1, n_init=50, random_state=1).fit(df[syms]).inertia_
for i in range(2, 15):
    km = KMeans(n_clusters=i, n_init=50, random_state=1)
    km.fit(df[syms])
    pct_var.append((total_ss - km.inertia_) / total_ss)

plt.figure(figsize=(5, 4))
plt.plot(range(2, 15), pct_var, 'o-')
plt.xlabel('Число кластеров')
plt.ylabel('% объясненной дисперсии')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'psds_0706.png'))
plt.close()


# ====== Иерархическая кластеризация ======

syms1 = ['GOOGL', 'AMZN', 'AAPL', 'MSFT', 'CSCO', 'INTC', 'CVX', 
         'XOM', 'SLB', 'COP', 'JPM', 'WFC', 'USB', 'AXP',
         'WMT', 'TGT', 'HD', 'COST']

df = sp500_px.loc[sp500_px.index >= '2011-01-01', syms1]
d = linkage(df.T, method='ward')

# Рисунок 7.7: Дендрограмма
plt.figure(figsize=(5, 5))
dendrogram(d, labels=df.columns, leaf_rotation=90)
plt.ylabel('Расстояние')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'psds_0707.png'))
plt.close()

# Рисунок 7.8: Сравнение методов кластеризации
df0 = sp500_px.loc[sp500_px.index >= '2011-01-01', ['XOM', 'CVX']]

methods = ['single', 'average', 'complete', 'ward']
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for i, method in enumerate(methods):
    d = linkage(df0, method=method)
    clusters = cut_tree(d, n_clusters=4).flatten()
    df0_plot = df0.copy()
    df0_plot['cluster'] = clusters
    
    sns.scatterplot(data=df0_plot, x='XOM', y='CVX', hue='cluster', 
                    style='cluster', ax=axes[i])
    axes[i].set_title(method)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'psds_0708.png'))
plt.close()


# ====== Модельная кластеризация ======

# Рисунок 7.9: Многомерные нормальные эллипсы
mu = np.array([0.5, -0.5])
sigma = np.array([[1, 1], [1, 2]])
prob = [0.5, 0.75, 0.95, 0.99]

fig, ax = plt.subplots(figsize=(5, 5))
for p in prob:
    # Эллипс для многомерного нормального распределения
    chi2_val = chi2.ppf(p, df=2)
    eigvals, eigvecs = np.linalg.eigh(sigma)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2 * np.sqrt(chi2_val * eigvals)
    ellipse = Ellipse(xy=mu, width=width, height=height,
                     angle=angle, alpha=0.3, color='grey')
    ax.add_patch(ellipse)
ax.scatter(mu[0], mu[1], s=100, color='black')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'psds_0709.png'))
plt.close()

# Рисунок 7.10: mclust для XOM и CVX
df = sp500_px.loc[sp500_px.index >= '2011-01-01', ['XOM', 'CVX']]

gmm = GaussianMixture(n_components=2, random_state=1)
df['cluster'] = gmm.fit_predict(df)

plt.figure(figsize=(6, 5))
sns.scatterplot(data=df, x='XOM', y='CVX', hue='cluster', style='cluster')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'psds_0710.png'))
plt.close()

# Рисунок 7.11: BIC scores
n_components_range = range(1, 10)
bic_scores = []
for n in n_components_range:
    gmm = GaussianMixture(n_components=n, random_state=1)
    gmm.fit(df[['XOM', 'CVX']])
    bic_scores.append(gmm.bic(df[['XOM', 'CVX']]))

plt.figure(figsize=(5, 5))
plt.plot(n_components_range, bic_scores, 'o-')
plt.xlabel('Число компонент')
plt.ylabel('BIC')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'psds_0711.png'))
plt.close()


# ====== Масштабирование ======

# Пример с данными о кредитах
loan_data = pd.read_csv(os.path.join(DATA_PATH, "loan_data.csv"))
defaults = loan_data[loan_data.outcome == 'default']
df = defaults[['loan_amnt', 'annual_inc', 'revol_bal', 'open_acc', 'dti', 'revol_util']]

km = KMeans(n_clusters=4, n_init=10, random_state=1)
df['cluster'] = km.fit_predict(df)

# Без масштабирования
centers = pd.DataFrame(km.cluster_centers_, columns=df.columns[:-1])
print("Центры кластеров без масштабирования:")
print(centers.round(2))

# С масштабированием
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[df.columns[:-1]])
km_scaled = KMeans(n_clusters=4, n_init=10, random_state=1)
df_scaled_clustered = km_scaled.fit_predict(df_scaled)

centers_scaled = pd.DataFrame(km_scaled.cluster_centers_, columns=df.columns[:-1])
# Обратное преобразование центров
centers_scaled_original = scaler.inverse_transform(centers_scaled)
centers_scaled_original = pd.DataFrame(centers_scaled_original, columns=df.columns[:-1])
print("\nЦентры кластеров с масштабированием:")
print(centers_scaled_original.round(2))


# ====== Категориальные данные и расстояние Говера ======

x = loan_data.iloc[:5][['dti', 'payment_inc_ratio', 'home_', 'purpose_']]
print("\nПример данных для расстояния Говера:")
print(x)

# Вычисление расстояния Говера
distance_matrix = gower.gower_matrix(x)
print("\nМатрица расстояний Говера:")
print(distance_matrix)

# Иерархическая кластеризация с расстоянием Говера
np.random.seed(301)
df = loan_data.sample(250)[['dti', 'payment_inc_ratio', 'home_', 'purpose_']]
distance_matrix = gower.gower_matrix(df)
d = linkage(distance_matrix, method='ward')

# Рисунок 7.13: Дендрограмма для категориальных данных
plt.figure(figsize=(5, 5))
dendrogram(d, leaf_rotation=90, leaf_font_size=8)
plt.ylabel('Расстояние')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'psds_0713.png'))
plt.close()

