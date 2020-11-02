from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.random_projection import SparseRandomProjection

from hw1.utils import Adult
from hw1.utils import Diabetes
from hw1.main import Processor as Processor3

datasets = [
    Adult(),
    Diabetes()
]


def visualize_kmeans_clusters():
    global pcadf
    for dataset in datasets:
        processor = Processor3()
        processor.latext_start_figure()
        X_train, X_test, y_train, y_test, _ = dataset.get_data(model='KMeans')
        n_clusters = len(dataset.label_encoder.classes_)
        clustering_models = [
            # KMeans(n_clusters=n_clusters,
            #        init='k-means++',
            #        n_init=10,
            #        max_iter=600,
            #        random_state=0,
            #        tol=0.0001),
            GaussianMixture(n_components=n_clusters,
                            n_init=10,
                            max_iter=600,
                            random_state=0,
                            tol=0.0001)
        ]

        for cluster in clustering_models:
            pipe = Pipeline(
                [
                    ("model", cluster)
                ]
            )

            if isinstance(pipe["model"], KMeans):
                X_r = pipe.fit_transform(X_train)
                pcadf = pd.DataFrame(X_r)
            elif isinstance(pipe["model"], GaussianMixture):
                pipe.fit(X_train)
                X_r = X_train
                pcadf = pd.DataFrame(X_r)

            predicted_labels = pipe.predict(X_train)

            if isinstance(pipe["model"], KMeans):
                loc_ = pcadf.loc[:, 0:1]
            elif isinstance(pipe["model"], GaussianMixture):
                loc_ = pcadf.loc[:, dataset.top_10_features_idx[0:2]]

            pcadf = pd.DataFrame(data=loc_, index=pcadf.index)
            pcadf.columns = ["component_1", "component_2"]
            pcadf["predicted_cluster"] = predicted_labels
            pcadf["true_label"] = dataset.label_encoder.inverse_transform(y_train)

            plt.style.use("fivethirtyeight")
            plt.figure(figsize=(12, 12))

            scat = sns.scatterplot(
                x="component_1",
                y="component_2",
                s=50,
                data=pcadf,
                hue="predicted_cluster",
                style="true_label",
                palette="Set1",
            )

            scat.set_title(f"{cluster.__class__.__name__}")
            plt.legend(loc='best')
            # plt.show()

            filename = '%s_%s_%s' % ('clusters_original', cluster.__class__.__name__, dataset.__class__.__name__)
            chart_path = 'report/images/%s.png' % filename
            plt.savefig(chart_path)
            plt.close()
            print(chart_path)
            processor.latex_subgraph(dataset=dataset.__class__.__name__, fig=filename, caption='',
                                     filename=filename)

        processor.latex_end_figure(caption=f"{dataset.__class__.__name__} Clusters",
                                   fig=f"original_{dataset.__class__.__name__}")


if __name__ == "__main__":
    visualize_kmeans_clusters()
