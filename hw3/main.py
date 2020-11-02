import argparse
import itertools
import os
import matplotlib.pyplot as plt
import scipy
from kneed import KneeLocator
from time import time
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib as mpl
from mlrose_hiive import NeuralNetwork
from numpy import linalg
from scipy.stats import kurtosis
from sklearn import random_projection
import scipy.sparse as sps
from scipy.linalg import pinv

from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import FastICA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
import seaborn as sns
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.random_projection import SparseRandomProjection

from hw1.main import Processor
from hw1.utils import Adult
from hw1.utils import Config
from hw1.utils import Diabetes

from hw1.main import Processor as P1
from hw3.feature_importance import run_feature_importance
from hw3.reductions import run_dimension_reductions
from hw3.run_vizualize_kmeans_clusters import visualize_kmeans_clusters


def title(input):
    out = [x[0].upper() + x[1:] for x in input.split('_')]
    return ''.join(out)


class Processor3(Processor):
    def __init__(self):
        super().__init__()


def compute_kmeans_elbow_curves(reduced_X=None, reduction=None):
    plt.figure()
    processor.latext_start_figure()
    for dataset in datasets:
        dataset_name = dataset.__class__.__name__
        print('%s' % dataset_name)
        X_train, X_test, y_train, y_test, _ = dataset.get_data(model='KMeans')
        if reduced_X:
            if reduction not in reduced_X.keys():
                return
            if dataset_name not in reduced_X.get(reduction).keys():
                return
            X_train = reduced_X.get(reduction).get(dataset_name)

        distortions = []
        clusters = []
        times = []
        iterations = []
        silhouette_coefficients = []
        ari_scores = []
        for x in range(2, 21):
            i = int(x)
            print('# of clusters: %i' % i)
            km = KMeans(n_clusters=i,
                        init='k-means++',
                        n_init=10,
                        max_iter=600,
                        random_state=0,
                        tol=0.0001)
            try:
                t0 = time()
                km.fit(X_train)
                times.append(round(time() - t0, 6))
                score = silhouette_score(X_train, km.labels_).round(2)
                clusters.append(i)
                distortions.append(km.inertia_)
                iterations.append(km.n_iter_)
                silhouette_coefficients.append(score)
                ari_kmeans = adjusted_rand_score(y_train, km.labels_)
                ari_scores.append(ari_kmeans)
            except Exception as e:
                print(e)

        draw_plot(clusters, distortions, 'Distortion', dataset_name, "kmeans", reduction)
        draw_plot(clusters, times, 'Training Time', dataset_name, "kmeans", reduction)
        draw_plot(clusters, iterations, 'Iterations', dataset_name, "kmeans", reduction)
        draw_plot(clusters, silhouette_coefficients, 'Silhouette Coefficient', dataset_name, "kmeans", reduction)
        draw_plot(clusters, ari_scores, 'Adjusted Rand Score', dataset_name, "kmeans", reduction)

        kl = KneeLocator(clusters, distortions, curve="convex", direction="decreasing")
        print(kl.elbow)
    processor.latex_end_figure(caption="Cluster Validation", fig="cluster_curve")


def compute_em_elbow_curves():
    plt.figure()
    processor.latext_start_figure()
    for dataset in datasets:
        dataset_name = dataset.__class__.__name__
        print('%s' % dataset_name)
        X_train, X_test, y_train, y_test, _ = dataset.get_data(model='KMeans')
        distortions = []
        clusters = []
        times = []
        iterations = []
        silhouette_coefficients = []
        aics = []
        for x in range(2, 11):
            i = int(x)
            print('# of clusters: %i' % i)
            km = GaussianMixture(n_components=i,
                                 n_init=10,
                                 max_iter=600,
                                 random_state=0,
                                 tol=0.0001)
            try:
                t0 = time()
                km.fit(X_train)
                times.append(round(time() - t0, 6))
                print('Converged:', km.converged_)  # Check if the model has converged
                means = km.means_
                covariances = km.covariances_
                aics.append(km.aic(X_train))
                distortions_score = km.score(X=X_train)
                distortions.append(1.0 / distortions_score)
                labels = km.predict(X=X_train)
                score = silhouette_score(X_train, labels)
                clusters.append(i)
                iterations.append(km.n_iter_)
                silhouette_coefficients.append(score)
            except Exception as e:
                pass

        draw_plot(clusters, distortions, 'Distortion', dataset_name, "em")
        draw_plot(clusters, aics, 'AIC', dataset_name, "em")
        draw_plot(clusters, times, 'Training Time', dataset_name, "em")
        draw_plot(clusters, iterations, 'Iterations', dataset_name, "em")
        draw_plot(clusters, silhouette_coefficients, 'Silhouette Coefficient', dataset_name, "em")

        kl = KneeLocator(clusters, distortions, curve="convex", direction="decreasing")
        print(kl.elbow)
    processor.latex_end_figure(caption="Cluster Validation", fig="cluster_curve")


def run_bics():
    for dataset in datasets:
        dataset_name = dataset.__class__.__name__
        X, X_test, y_train, y_test, target_names = dataset.get_data(model='KMeans')
        lowest_bic = np.infty
        bic = []
        n_components_range = range(1, 16)
        cv_types = ['spherical', 'tied', 'diag', 'full']
        for cv_type in cv_types:
            for n_components in n_components_range:
                # Fit a Gaussian mixture with EM
                gmm = GaussianMixture(n_components=n_components, covariance_type=cv_type)
                gmm.fit(X)
                bic.append(gmm.bic(X))
                if bic[-1] < lowest_bic:
                    lowest_bic = bic[-1]
                    best_gmm = gmm

        bic = np.array(bic)
        color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                                      'darkorange'])
        clf = best_gmm
        bars = []

        # Plot the BIC scores
        plt.figure(figsize=(8, 6))
        spl = plt.subplot(1, 1, 1)
        for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
            xpos = np.array(n_components_range) + .2 * (i - 2)
            bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                          (i + 1) * len(n_components_range)],
                                width=.2, color=color))
        plt.xticks(n_components_range)
        plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
        plt.title('BIC score per model')
        xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 + \
               .2 * np.floor(bic.argmin() / len(n_components_range))
        plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
        spl.set_xlabel('Number of components')
        spl.legend([b[0] for b in bars], cv_types)
        filename = '%s_%s_%s' % ('em', 'bics', dataset_name)
        chart_path = 'report/images/%s.png' % filename
        plt.savefig(chart_path)
        plt.close()
        processor.latex_subgraph(dataset=dataset_name, fig=filename, caption=dataset_name,
                                 filename=filename)


def run_pca():
    plt.figure()
    processor.latext_start_figure()
    for dataset in datasets:
        dataset_name = dataset.__class__.__name__
        print('%s' % dataset_name)
        X_train, X_test, y_train, y_test, target_names = dataset.get_data(model='KMeans')
        pca = PCA(n_components=2)
        X_r = pca.fit(X_train).transform(X_train)

        print('explained variance ratio (first two components): %s'
              % str(pca.explained_variance_ratio_))

        colors = ['navy', 'turquoise']
        lw = 2

        for color, i, target_name in zip(colors, [0, 1], target_names):
            plt.scatter(X_r[y_train == i, 0], X_r[y_train == i, 1], color=color, alpha=.8, lw=lw,
                        label=target_name)
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.title('PCA of %s' % dataset_name)
        filename = '%s_%s' % ('pca', dataset_name)
        chart_path = 'report/images/%s.png' % filename
        plt.savefig(chart_path)
        plt.close()
        processor.latex_subgraph(dataset=dataset_name, fig=filename, caption=dataset_name,
                                 filename=filename)

        lr = LogisticRegression()
        lr.fit(X_r, y_train)
        plot_decision_regions(X_r, y_train, classifier=lr)
        plt.xlabel('PC 1')
        plt.xlabel('PC 2')
        plt.legend(loc='lower left')
        plt.show()

    processor.latex_end_figure(caption="PCA Charts", fig="pca_curve")


def run_ica():
    plt.figure()
    processor.latext_start_figure()
    for dataset in datasets:
        dataset_name = dataset.__class__.__name__
        print('%s' % dataset_name)
        X_train, X_test, y_train, y_test, target_names = dataset.get_data(model='KMeans')
        pca = FastICA(n_components=2, random_state=0)
        X_r = pca.fit(X_train).transform(X_train)
        colors = ['navy', 'turquoise']
        lw = 2

        for color, i, target_name in zip(colors, [0, 1], target_names):
            plt.scatter(X_r[y_train == i, 0], X_r[y_train == i, 1], color=color, alpha=.8, lw=lw,
                        label=target_name)
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.title('ICA of %s' % dataset_name)
        filename = '%s_%s' % ('ica', dataset_name)
        chart_path = 'report/images/%s.png' % filename
        plt.savefig(chart_path)
        plt.close()
        processor.latex_subgraph(dataset=dataset_name, fig=filename, caption=dataset_name,
                                 filename=filename)

        lr = LogisticRegression()
        lr.fit(X_r, y_train)
        plot_decision_regions(X_r, y_train, classifier=lr)
        plt.xlabel('PC 1')
        plt.xlabel('PC 2')
        plt.legend(loc='lower left')
        plt.show()

        fastICAComponents = pca.components_
        kurtS = scipy.stats.kurtosis(fastICAComponents, axis=1)
        kurtIdx = np.argmax(kurtS)
        # icaFeatures[i, :] = fastICAComponents[kurtIdx, :]
        print(2, 'Kurtosis: ' + str(kurtS[kurtIdx]), 'S Shape: ' + str(X_r.shape), 'Kurt Shape: ' + str(kurtS.shape),
              'ICA Shape: ' + str(fastICAComponents.shape))
        n = kurtosis(X_r)
        abs_n = abs(n)
        k = sum(abs_n)
        print("done")

    processor.latex_end_figure(caption="ICA Charts", fig="ica_curve")


def run_ica_v2():
    for dataset in datasets:
        dataset_name = dataset.__class__.__name__
        print('%s' % dataset_name)
        X_train, X_test, y_train, y_test, _ = dataset.get_data(model='KMeans')
        kurtosisses = []
        kurtosisses_ = []
        clusters = []
        steps = 1 if dataset_name == 'Diabetes' else 10
        maxsteps = 11 if dataset_name == 'Diabetes' else 101
        for x in range(2, maxsteps, steps):
            pca = FastICA(n_components=x, random_state=0)
            X_r = pca.fit(X_train).transform(X_train)
            X = X_r.T
            kurts = []
            for i in range(X.shape[0]):
                values = X[i, :]
                kurt_ = round(kurtosis(values), 4)
                kurts.append(kurt_)
            tmp = pd.DataFrame(X_r)
            tmp = tmp.kurt(axis=0)
            mean = tmp.abs().mean()
            kurtosisses.append(mean)
            n = kurtosis(np.mean(kurts))
            kurtosisses_.append(n)
            clusters.append(x)

        draw_plot(clusters, kurtosisses, 'Kurtosis', dataset_name, "ica")
        draw_plot(clusters, kurtosisses_, 'Kurtosis_', dataset_name, "ica")


def draw_plot(x_vals, y_vals, y_axis, dataset_name, algo, reduction=None):
    title = y_axis.replace(" ", "_").lower()
    filename = '%s_%s_%s' % (algo, title, dataset_name)
    filename = filename if not reduction else f'{filename}_{reduction}'

    chart_path = 'report/images/%s.png' % filename
    plt.plot(x_vals, y_vals)
    plt.xlabel('Number of clusters')
    plt.ylabel(y_axis)
    plt.savefig(chart_path)
    plt.close()
    processor.latex_subgraph(dataset=dataset_name, fig=filename, caption=dataset_name,
                             filename=filename)


def draw_plots(x_vals, y_vals, y_axis, dataset_name, algo, labels, reduction=None):
    title = y_axis.replace(" ", "_").lower()
    filename = '%s_%s_%s' % (algo, title, dataset_name)
    filename = filename if not reduction else f'{filename}_{reduction}'
    chart_path = 'report/images/%s.png' % filename
    for y_val, y_label in zip(y_vals, labels):
        plt.plot(x_vals, y_val, label=y_label)
    plt.xlabel('Number of clusters')
    plt.xticks(x_vals, x_vals)
    plt.ylabel(y_axis)
    plt.legend()
    plt.savefig(chart_path)
    plt.close()
    processor.latex_subgraph(dataset=dataset_name, fig=filename, caption=dataset_name,
                             filename=filename)


def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors=colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.6,
                    c=cmap(idx),
                    edgecolors='black',
                    marker=markers[idx],
                    label=cl)


def plot_heatmaps():
    for dataset in datasets:
        sns.pairplot(dataset.df)
        plt.tight_layout()
        plt.show()


def plot_pca_variance():
    plt.figure()
    processor.latext_start_figure()
    for dataset in datasets:
        X_train, X_test, y_train, y_test, target_names = dataset.get_data(model='KMeans')
        pca = PCA(n_components=None)
        pca.fit(X_train)
        eigen_vals = pca.explained_variance_ratio_
        tot = sum(eigen_vals)
        var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
        cum_var_exp = np.cumsum(var_exp)
        plt.bar(range(1, len(eigen_vals) + 1), var_exp, alpha=0.5, align='center',
                label='individual explained variance')
        plt.step(range(1, len(eigen_vals) + 1), cum_var_exp, where='mid', label='cumulative explained variance')
        plt.xlabel("Component")
        plt.ylabel("Covariance")
        dataset_name = dataset.__class__.__name__
        filename = '%s_%s_%s' % ('pca', 'cov', dataset_name)
        chart_path = 'report/images/%s.png' % filename
        plt.savefig(chart_path)
        plt.close()
        processor.latex_subgraph(dataset=dataset_name, fig=filename, caption=dataset_name,
                                 filename=filename)
    processor.latex_end_figure(caption="Covariance Distribution", fig="pca_cov_distribution")


def plot_lda_variance():
    plt.figure()
    processor.latext_start_figure()
    for dataset in datasets:
        X_train, X_test, y_train, y_test, target_names = dataset.get_data(model='KMeans')
        pca = LinearDiscriminantAnalysis(n_components=None, store_covariance=True)
        pca.fit(X_train, y_train)
        eigen_vals = pca.explained_variance_ratio_
        tot = sum(eigen_vals)
        var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
        cum_var_exp = np.cumsum(var_exp)
        plt.bar(range(1, len(eigen_vals) + 1), var_exp, alpha=0.5, align='center',
                label='individual explained variance')
        plt.step(range(1, len(eigen_vals) + 1), cum_var_exp, where='mid', label='cumulative explained variance')
        plt.xlabel("Component")
        plt.ylabel("Covariance")
        dataset_name = dataset.__class__.__name__
        filename = '%s_%s_%s' % ('pca', 'cov', dataset_name)
        chart_path = 'report/images/%s.png' % filename
        plt.savefig(chart_path)
        plt.close()
        processor.latex_subgraph(dataset=dataset_name, fig=filename, caption=dataset_name,
                                 filename=filename)
    processor.latex_end_figure(caption="Covariance Distribution", fig="pca_cov_distribution")


def plot_kernel_pca_variance():
    for dataset in datasets:
        X_train, X_test, y_train, y_test, target_names = dataset.get_data(model='KMeans')
        pca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
        pca.fit(X_train)
        X_skernpca = pca.transform(X_train)
        plt.scatter(X_skernpca[y_train == 0, 0], X_skernpca[y_train == 0, 1],
                    color='red', marker='^', alpha=0.5)
        plt.scatter(X_skernpca[y_train == 1, 0], X_skernpca[y_train == 1, 1],
                    color='blue', marker='o', alpha=0.5)
        plt.show()


def reconstructionError(projections, X):
    W = projections.components_
    if sps.issparse(W):
        W = W.todense()
    p = pinv(W)
    reconstructed = ((p @ W) @ (X.T)).T  # Unproject projected data
    errors = np.square(X - reconstructed)
    return np.nanmean(errors)


def run_rp():
    for dataset in datasets:
        X_train, X_test, y_train, y_test, target_names = dataset.get_data(model='KMeans')
        errors = []
        clusters = []
        dataset_name = dataset.__class__.__name__
        steps = 1 if dataset_name == 'Diabetes' else 10
        maxsteps = 11 if dataset_name == 'Diabetes' else 101
        for x in range(2, maxsteps, steps):
            transformer = SparseRandomProjection(random_state=0, eps=0.999999, n_components=x)
            X_new = transformer.fit_transform(X_train)
            error = reconstructionError(transformer, X_train)
            print(X_new.shape)
            print(error)
            errors.append(error)
            clusters.append(x)

        draw_plot(clusters, errors, 'Reconstruction Error', dataset_name, "rp")


def run_lda():
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    processor = P1()
    datasets = [Diabetes(), Adult()]
    estimators = [
        Config(name='lda',
               estimator=LinearDiscriminantAnalysis(),
               cv=kfold,
               params={
               })]

    for dataset in datasets:
        for estimator in estimators:
            estimator = processor.get_default_model(dataset=dataset, estimator=estimator)
            processor.process_validations(dataset=dataset, estimator=estimator)
            processor.plot_validation()

    for dataset in datasets:
        for estimator in estimators:
            estimator = processor.get_default_model(dataset=dataset, estimator=estimator)
            processor.param_selection(dataset=dataset, estimator=estimator)
            processor.print_best_params()

    for dataset in datasets:
        for estimator in estimators:
            processor.process(dataset=dataset, estimator=estimator)
            processor.plot_learning_curves()


def run_em_again(reduced_X=None, reduction=None):
    for dataset in datasets:
        dataset_name = dataset.__class__.__name__
        steps = 100 if dataset_name == 'Diabetes' else 1
        maxsteps = 801 if dataset_name == 'Diabetes' else 21
        n_components = np.arange(1, maxsteps, steps)
        for i in range(1):
            X, X_test, y_train, y_test, target_names = dataset.get_data(model='KMeans')
            if reduced_X:
                if reduction not in reduced_X.keys():
                    return
                if dataset_name not in reduced_X.get(reduction).keys():
                    return
                X = reduced_X.get(reduction).get(dataset_name)

            try:
                models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(X) for n in n_components]
            except:
                raise
            draw_plots(n_components, [[m.bic(X) for m in models], [m.aic(X) for m in models]], 'Performance',
                       dataset_name, "em", labels=['BIC', 'AIC'], reduction=reduction)


def run_16_iters():
    models = [
        PCA(n_components=2, random_state=0),
        FastICA(n_components=2, random_state=0),
        LinearDiscriminantAnalysis(n_components=1),
        SparseRandomProjection(random_state=0, n_components=2)
    ]
    configs = {}
    for model in models:
        configs[model.__class__.__name__.lower()] = {}

    for dataset in datasets:
        dataset_name = dataset.__class__.__name__
        X_train, X_test, y_train, y_test, target_names = dataset.get_data(model='KMeans')

        for model in models:
            reduction = model.__class__.__name__.lower()
            print(reduction)
            if reduction == 'lineardiscriminantanalysis':
                X_r = model.fit_transform(X_train, y_train)
            else:
                X_r = model.fit_transform(X_train)
            configs[reduction][dataset_name] = X_r
            compute_kmeans_elbow_curves(reduced_X=configs, reduction=reduction)
            run_em_again(reduced_X=configs, reduction=reduction)


def run_nn_opt():
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    processor = P1()
    datasets = [
        Diabetes()
    ]
    estimators = [
        Config(name='NN_%s' % title('pca'),
               estimator=Pipeline([('pca', PCA(n_components=7, random_state=0)), ('nn', MLPClassifier(max_iter=1000))]),
               cv=kfold,
               params={}),
        Config(name='NN_%s' % title('ica'),
               estimator=Pipeline([('pca', FastICA(n_components=7, random_state=0)), ('nn', MLPClassifier(max_iter=1000))]),
               cv=kfold,
               params={}),
        Config(name='NN_%s' % title('rp'),
               estimator=Pipeline(
                   [('pca', SparseRandomProjection(random_state=0, n_components=7)), ('nn', MLPClassifier(max_iter=1000))]),
               cv=kfold,
               params={}),
        Config(name='NN_%s' % title('dl'),
               estimator=Pipeline([('pca', MiniBatchDictionaryLearning(n_components=7, batch_size=200, n_iter=1000, random_state=0)), ('nn', MLPClassifier(max_iter=1000))]),
               cv=kfold,
               params={})
    ]

    for dataset in datasets:
        for estimator in estimators:
            processor.process(dataset=dataset, estimator=estimator, control=True)
            processor.plot_learning_curves(control=True)


def run_nn_opt_clusters():
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    processor = P1()
    datasets = [
        Diabetes()
    ]

    dr_models = [
        PCA(n_components=7, random_state=0),
        FastICA(n_components=7, random_state=0),
        MiniBatchDictionaryLearning(n_components=7, batch_size=200, n_iter=1000, random_state=0),
        SparseRandomProjection(random_state=0, n_components=7)
    ]

    clustering_models = []
    for i in [2]:
        clustering_models.append(KMeans(n_clusters=i,
                                        init='k-means++',
                                        n_init=10,
                                        max_iter=600,
                                        random_state=0,
                                        tol=0.0001))
        clustering_models.append(GaussianMixture(n_components=i,
                                                 n_init=10,
                                                 max_iter=600,
                                                 random_state=0,
                                                 tol=0.0001))

    configs = {}
    for model in dr_models:
        configs[model.__class__.__name__.lower()] = {}

    for dataset in datasets:
        dataset_name = dataset.__class__.__name__
        X_train, X_test, y_train, y_test, target_names = dataset.get_data(model='KMeans')

        for model in dr_models:
            reduction = model.__class__.__name__.lower()
            print(reduction)
            if reduction == 'lineardiscriminantanalysis':
                X_r = model.fit_transform(X_train, y_train)
                X_r_test = model.transform(X_test)
            else:
                X_r = model.fit_transform(X_train)
                X_r_test = model.transform(X_test)

            for clustering_model in clustering_models:
                try:
                    clusters = clustering_model.n_components
                except:
                    clusters = clustering_model.n_clusters

                preds = clustering_model.fit_predict(X_train)
                preds_test = clustering_model.predict(X_test)
                df = pd.DataFrame(X_r)
                df['cluster'] = preds
                df_test = pd.DataFrame(X_r_test)
                df_test['cluster'] = preds_test
                print('done')
                data = {
                    'X_train': df,
                    'X_test': df_test,
                    'y_train': y_train,
                    'y_test': y_test
                }

                estimator = Config(
                    name='NN_%s_%s_%i' % (clustering_model.__class__.__name__.lower(), reduction, clusters),
                    estimator=Pipeline([('nn', MLPClassifier(max_iter=1000))]),
                    cv=kfold,
                    params={
                        'nn__restarts': [10]
                    })

                processor.process(dataset=dataset, estimator=estimator, control=True, data=data)
                processor.plot_learning_curves(control=True, suffix='_part5')


if __name__ == "__main__":
    try:
        os.makedirs('report/images', exist_ok=True)
        print("Directory created successfully.")
    except OSError as error:
        print("Directory '%s' can not be created")

    parser = argparse.ArgumentParser(description='Find X Coding Quiz')

    parser.add_argument('-m', '--mode', help='Mode', default='debug', dest='mode')
    args = parser.parse_args()

    processor = Processor3()

    datasets = [
        Diabetes(),
        Adult(),
    ]

    run_feature_importance()
    compute_kmeans_elbow_curves()
    visualize_kmeans_clusters()
    run_dimension_reductions()
    run_nn_opt()
    run_nn_opt_clusters()

    ## TODO: Uncomment to get the other charts

    compute_kmeans_elbow_curves()
    compute_em_elbow_curves()
    run_bics()
    plot_pca_variance()
    plot_kernel_pca_variance()
    plot_heatmaps()
    run_pca()
    run_ica()
    run_ica_v2()
    run_rp()
    plot_lda_variance()
    run_lda()
    run_em_again()
