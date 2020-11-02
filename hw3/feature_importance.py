import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set a seed for the random number generator for reproducibility
from sklearn.ensemble import RandomForestClassifier

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


def run_feature_importance():
    processor = Processor3()
    processor.latext_start_figure()
    for dataset in [
        Diabetes(),
        Adult()
    ]:
        processor = Processor3()
        processor.latext_start_figure()
        X_train, X_test, y_train, y_test, _ = dataset.get_data(model='KMeans')
        forest = RandomForestClassifier(n_estimators=500, random_state=1)
        forest.fit(X_train, y_train)
        importances = forest.feature_importances_

        indices = np.argsort(importances)[::-1]
        top_10 = []
        top_10_vals = []
        top_10_idx = []
        for f, g in zip(range(X_train.shape[1]), indices[:10]):
            print("%2d) % -*s %f" % (f+1, 30, dataset.fields[indices[f]], importances[indices[f]]))
            top_10.append(dataset.fields[indices[f]])
            top_10_idx.append(indices[f])
            top_10_vals.append(importances[indices[f]])

        print(top_10)
        print(top_10_idx)

        plt.title('Feature Importance')
        plt.bar(top_10, top_10_vals, align='center')
        plt.xticks(top_10, rotation=90)
        plt.tight_layout()
        # plt.show()
        filename = '%s_%s' % ('features', dataset.__class__.__name__)
        chart_path = 'report/images/%s.png' % filename
        plt.savefig(chart_path)
        plt.close()
        processor.latex_subgraph(dataset=dataset.__class__.__name__, fig=filename, caption=dataset.__class__.__name__, filename=filename)

    processor.latex_end_figure(caption=f"Feature Importance", fig=f"feature_importance")


if __name__ == "__main__":
    run_feature_importance()
