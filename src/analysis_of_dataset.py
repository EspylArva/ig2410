import pandas as pd
from scipy.stats import pearsonr
import numpy as np
from sklearn import decomposition
import matplotlib.pyplot as plt


def plotRaw(dataset, fig_size = 6):
    f, axes = plt.subplots()
    f.set_figheight(fig_size)
    f.set_figwidth(fig_size)

    axes.set_title('Raw Data', fontsize=20)

    sexes = np.sort(np.unique(dataset['sex']))
    chronics = np.sort(np.unique(dataset['chronic_disease_binary']))
    colors = ['b', 'r']
    legends = ['male', 'female']
    legends_chr = ['male with chronic disease', 'female with chronic disease']

    for sex, color, legend in zip(sexes, colors, legends):
        sex_ = dataset['sex'] == sex
        chronics_ = dataset['chronic_disease_binary'] == chronics[0]
        cond = sex_ & chronics_
        axes.scatter(dataset.loc[cond, 'country'], dataset.loc[cond, 'age'], c=color, s=50, label=legend, alpha=0.3)

    for sex, color, legend in zip(sexes, colors, legends_chr):
        sex_ = dataset['sex'] == sex
        chronics_ = dataset['chronic_disease_binary'] == chronics[1]
        cond = sex_ & chronics_
        axes.scatter(dataset.loc[cond, 'country'], dataset.loc[cond, 'age'], c=color, s=50, label=legend, alpha=0.3,
                     marker='x')
    plt.xticks(rotation=90)

    axes.set_xlabel('Countries')
    axes.set_ylabel('Age')
    axes.legend(legends + legends_chr, loc=1)
    axes.grid()



"""
    input: dataset
    output: correlations between every given parameter
    --
    Computing Pearson correlation on data
    https://realpython.com/numpy-scipy-pandas-correlation-python/
"""


def computeCorrelations(df):
    xy_corr = []
    for x in df.columns:
        x_corr = []
        for y in df.columns:
            corr_xy = pearsonr(df[x], df[y])[0]
            x_corr.append(corr_xy)
        xy_corr.append(x_corr)

    return pd.DataFrame(xy_corr, index=df.columns, columns=df.columns)


"""
    input: dataset, number_of_principal_components, size_of_figure
    output: plot of the PCA of the dataset
    --
    Computing the PCA for the given dataset. The PCA  works on the following columns (deemed to be more interesting):
    { 'age', 'sex', 'country', 'chronic_disease_binary' }
    --
    PCA: https://emanuelfontelles.github.io/blog/Principal-Component-Analysis.html
    Correlation vectors: https://stackoverflow.com/questions/39216897/plot-pca-loadings-and-loading-in-biplot-in-sklearn-like-rs-autoplot
"""


def pca(dataset, n_components=2, fig_size=8):
    pca_dataset = dataset.get(['age', 'sex', 'country', 'chronic_disease_binary']).copy()
    pca_ = decomposition.PCA(n_components=n_components)
    X_pca = pca_.fit_transform(pca_dataset)

    principalDf = pd.DataFrame(data=X_pca, columns=['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, dataset['outcome']], axis=1)

    explained_var = np.round(pca_.explained_variance_ratio_ * 100, decimals=2)

    f, axes = plt.subplots(1, 2)
    f.set_figheight(fig_size)
    f.set_figwidth(2*fig_size)

    axes[0].bar(x=range(len(explained_var)), height=explained_var, width=0.1, tick_label=['PC 1', 'PC2'])

    axes[1].set_title('2 component PCA', fontsize=20)
    targets = np.sort(np.unique(dataset['outcome']))
    colors = ['g', 'r']
    legends = ['alive', 'dead']
    for target, color, legend in zip(targets, colors, legends):
        indicesToKeep = finalDf['outcome'] == target
        axes[1].scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
                     finalDf.loc[indicesToKeep, 'principal component 2'], c=color, s=50, label=legend, alpha=0.3)
    axes[1].set_xlabel('PC1 - {0}%'.format(explained_var[0]))
    axes[1].set_ylabel('PC2 - {0}%'.format(explained_var[1]))
    axes[1].legend(legends, loc=1)
    axes[1].grid()

    for i, (x, y) in enumerate(zip(pca_.components_[0, :], pca_.components_[1, :])):
        axes[1].arrow(0, 0, x, y, color='black', head_width=0.05, head_length=0.05)
        axes[1].text(x + 0.1, y, pca_dataset.columns[i], fontsize='9', weight="bold", ha="center")

    plt.draw()
    return X_pca
