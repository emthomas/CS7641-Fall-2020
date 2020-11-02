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
from hw3.main import Processor3

datasets = [
    Adult(),
    Diabetes()
]

for dataset in datasets:
    processor = Processor3()
    processor.latext_start_figure()
    X_train, X_test, y_train, y_test, _ = dataset.get_data(model='KMeans')
    n_clusters = len(dataset.label_encoder.classes_)
    pca = PCA(n_components=0.95)
    pca.fit(X_train)
    n_components = pca.components_.shape[0]
    print(f"n_components: {n_components}")
    dr_models = [
        PCA(n_components=n_components, random_state=0),
        FastICA(n_components=n_components, random_state=0),
        MiniBatchDictionaryLearning(
            n_components=n_components, alpha=1, batch_size=200,
            n_iter=10, random_state=0),
        SparseRandomProjection(random_state=0, n_components=n_components)
    ]
    clustering_models = [
        KMeans(n_clusters=n_clusters,
               init='k-means++',
               n_init=10,
               max_iter=600,
               random_state=0,
               tol=0.0001),
        GaussianMixture(n_components=n_clusters,
                        n_init=10,
                        max_iter=600,
                        random_state=0,
                        tol=0.0001)
    ]
    for pca in dr_models:
        for cluster in clustering_models:
            preprocessor = Pipeline(
                [
                    ("reducer", pca),
                ]
            )

            clusterer = Pipeline(
                [
                    (
                        "model", cluster,
                    ),
                ]
            )

            pipe = Pipeline(
                [
                    ("preprocessor", preprocessor),
                    ("clusterer", clusterer)
                ]
            )

            pipe.fit(X_train)

            preprocessed_data = pipe["preprocessor"].transform(X_train)

            if isinstance(pipe["clusterer"]["model"], GaussianMixture):
                predicted_labels = pipe["clusterer"]["model"].predict(preprocessed_data)
            else:
                predicted_labels = pipe["clusterer"]["model"].labels_

            silhouette_score_val = silhouette_score(preprocessed_data, predicted_labels)
            print(f'Silhouette Score: {silhouette_score_val}')

            adjusted_rand_score_val = adjusted_rand_score(y_train, predicted_labels)
            print(f'ARI Score: {adjusted_rand_score_val}')

            pcadf = pd.DataFrame(
                pipe["preprocessor"].transform(X_train)
            )

            loc_ = pcadf.loc[:, 0:1]
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

            scat.set_title(f"{cluster.__class__.__name__} with {pca.__class__.__name__}")
            plt.legend(loc='best')

            filename = '%s_%s_%s_%s' % ('clusters', cluster.__class__.__name__, pca.__class__.__name__, dataset.__class__.__name__)
            chart_path = 'report/images/%s.png' % filename
            plt.savefig(chart_path)
            plt.close()
            processor.latex_subgraph(dataset=dataset.__class__.__name__, fig=filename, caption='', filename=filename)

    processor.latex_end_figure(caption=f"{dataset.__class__.__name__} Clusters Reduced",
                               fig=f"rc_{dataset.__class__.__name__}")