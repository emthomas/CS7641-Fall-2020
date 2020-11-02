import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set a seed for the random number generator for reproducibility
from hw1.main import Processor as Processor3

np.random.seed(0)
from sklearn.decomposition import FastICA
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.random_projection import SparseRandomProjection

from hw1.utils import Adult
from hw1.utils import Diabetes


def anomalyScores(originalDF, reducedDF):
    loss = np.sum((np.array(originalDF) - np.array(reducedDF)) ** 2, axis=1)
    loss = pd.Series(data=loss, index=originalDF.index)
    loss = (loss - np.min(loss)) / (np.max(loss) - np.min(loss))
    return loss


def scatterPlot(xDF, yDF, algoName):
    sns.set_theme(style="ticks")
    loc_ = xDF.loc[:, 0:1]
    tempDF = pd.DataFrame(data=loc_, index=xDF.index)
    tempDF = pd.concat((tempDF, yDF), axis=1, join="inner")
    tempDF.columns = ["Component 1", "Component 2", "Label"]
    sns.lmplot(x="Component 1", y="Component 2", hue="Label", data=tempDF, fit_reg=False)
    ax = plt.gca()
    ax.set_title("Separation of Observations using " + algoName)


def plotResults(trueLabels, anomalyScores, returnPreds=False, algo='', dataset_name='', average_error=None):
    plt.figure()
    preds = pd.concat([trueLabels, anomalyScores], axis=1)
    preds.columns = ['trueLabel', 'anomalyScore']
    precision, recall, thresholds = \
        precision_recall_curve(preds['trueLabel'], preds['anomalyScore'])
    average_precision = \
        average_precision_score(preds['trueLabel'], preds['anomalyScore'])

    plt.step(recall, precision, color='k', alpha=0.7, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.3, color='k')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Mean Precision: {average_precision:0.2f}; Mean Reconstruction Error: {average_error:0.2f}')

    title = algo.replace(" ", "_").lower()
    filename = '%s_%s_%s' % ('recall', title, dataset_name)
    chart_path = 'report/images/%s.png' % filename
    plt.savefig(chart_path)
    plt.close()
    processor = Processor3()
    replace = algo.replace('MiniBatch', '').replace('Sparse', '')
    processor.latex_subgraph(dataset=dataset_name, fig=filename, caption=replace, filename=filename)

    if returnPreds == True:
        return preds


from sklearn.decomposition import PCA


def run_dimension_reductions():
    global mean
    for dataset in [
        Diabetes(),
        Adult()
    ]:
        processor = Processor3()
        processor.latext_start_figure()
        X_train, X_test, y_train, y_test, _ = dataset.get_data(model='KMeans')
        pca = PCA(n_components=0.95)
        pca.fit(X_train)
        n_components = pca.components_.shape[0]
        print(f"n_components: {n_components}")

        whiten = True
        random_state = 0
        dr_models = [
            PCA(n_components=n_components, random_state=0),
            FastICA(n_components=n_components, random_state=0),
            MiniBatchDictionaryLearning(
                n_components=n_components, alpha=1, batch_size=200,
                n_iter=10, random_state=random_state),
            SparseRandomProjection(random_state=0, n_components=n_components)
        ]
        for pca in dr_models:
            X_train = pd.DataFrame(X_train)
            y_train = pd.DataFrame(y_train)

            if isinstance(pca, SparseRandomProjection):
                X_train_PCA = pca.fit_transform(X_train)
                X_train_PCA = pd.DataFrame(data=X_train_PCA, index=X_train.index)
                X_train_PCA_inverse = np.array(X_train_PCA).dot(pca.components_.todense())
                X_train_PCA_inverse = pd.DataFrame(data=X_train_PCA_inverse, index=X_train.index)
                scatterPlot(X_train_PCA, y_train, pca.__class__.__name__)
            elif isinstance(pca, MiniBatchDictionaryLearning):
                X_train_PCA = pca.fit_transform(X_train)
                X_train_PCA = pd.DataFrame(data=X_train_PCA, index=X_train.index)
                X_train_PCA_inverse = np.array(X_train_PCA).dot(pca.components_) + np.array(X_train.mean(axis=0))
                X_train_PCA_inverse = pd.DataFrame(data=X_train_PCA_inverse, index=X_train.index)
                scatterPlot(X_train_PCA, y_train, pca.__class__.__name__)
            else:
                X_train_PCA = pca.fit_transform(X_train)
                X_train_PCA = pd.DataFrame(data=X_train_PCA, index=X_train.index)
                X_train_PCA_inverse = pca.inverse_transform(X_train_PCA)
                X_train_PCA_inverse = pd.DataFrame(data=X_train_PCA_inverse, index=X_train.index)
                scatterPlot(X_train_PCA, y_train, pca.__class__.__name__)

            # plt.show()

            anomalyScoresPCA = anomalyScores(X_train, X_train_PCA_inverse)
            mean = np.mean(anomalyScoresPCA)
            print(mean)
            preds = plotResults(y_train, anomalyScoresPCA, True, pca.__class__.__name__, dataset.__class__.__name__,
                                mean)
        processor.latex_end_figure(caption=f"{dataset.__class__.__name__} Precision-Recall Curve",
                                   fig=f"pr_{dataset.__class__.__name__}")


if __name__ == "__main__":
    run_dimension_reductions()
