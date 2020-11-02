import argparse
import itertools
import random as r

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from hw1.utils import Adult
from hw1.utils import Config
from hw1.utils import Diabetes

r.seed(1)
np.random.seed(31415)

LOG_SCALES = {'max_depth': 2, 'C': 10, 'max_iter': 2, 'hidden_layer_sizes': 2}


def get_y_limits(train_mean, train_std, test_mean, test_std):
    y_max = np.max(train_mean + train_std) + 0.01
    y_min = np.min(test_mean - test_std) - 0.01
    return [y_min, y_max]


def get_one_layers(dataset=None):
    param = []
    for layers in range(1):
        layer = 2 ** layers
        for x in itertools.product((1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024), repeat=layer):
            param.append(x)
    return param


def get_multi_layers(dataset=None):
    param = []
    for i in range(10):
        out = []
        for j in range(i):
            out.append(32)
        param.append(tuple(out))
    return param[1:]


class Processor(object):
    def __init__(self):
        self.metrics = None
        self.initial_metrics = None
        self.best_models = {}
        self.best_params = {}

    def process(self, dataset, estimator, control=False, data=None):
        print("Processing Learning Curves for %s with %s." % (dataset.__class__.__name__, estimator.name))
        model = self.best_models.get(
            dataset.__class__.__name__ + "_" + estimator.name) if not control else estimator.estimator

        if not data:
            X_train, X_test, y_train, y_test = dataset.get_data(model=estimator.name)
        else:
            X_train, X_test, y_train, y_test = data.get('X_train'), data.get('X_test'), data.get('y_train'), data.get('y_test')

        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(estimator=model,
                                                                              X=X_train,
                                                                              y=y_train,
                                                                              train_sizes=np.linspace(0.1, 1.0, 10),
                                                                              n_jobs=-1,
                                                                              return_times=True,
                                                                              cv=estimator.cv)

        if not control:
            if not self.metrics:
                self.metrics = {}
            if dataset.__class__.__name__ not in self.metrics.keys():
                self.metrics[dataset.__class__.__name__] = {
                    'train_mean': {},
                    'train_std': {},
                    'test_mean': {},
                    'test_std': {},
                    'fit_times_mean': {},
                    'fit_times_std': {},
                    'train_sizes': {},
                    'X_test': {},
                    'y_test': {},
                    'results': [],
                    'names': [],
                    'validations': []
                }

            self.metrics[dataset.__class__.__name__]['train_mean'][estimator.name] = np.mean(train_scores, axis=1)
            self.metrics[dataset.__class__.__name__]['train_std'][estimator.name] = np.std(train_scores, axis=1)
            self.metrics[dataset.__class__.__name__]['test_mean'][estimator.name] = np.mean(test_scores, axis=1)
            self.metrics[dataset.__class__.__name__]['test_std'][estimator.name] = np.std(test_scores, axis=1)
            self.metrics[dataset.__class__.__name__]['fit_times_mean'][estimator.name] = np.mean(fit_times, axis=1)
            self.metrics[dataset.__class__.__name__]['fit_times_std'][estimator.name] = np.std(fit_times, axis=1)
            self.metrics[dataset.__class__.__name__]['train_sizes'] = train_sizes
            self.metrics[dataset.__class__.__name__]['X_test'] = X_test
            self.metrics[dataset.__class__.__name__]['y_test'] = y_test
            self.metrics[dataset.__class__.__name__]['X_train'] = X_train
            self.metrics[dataset.__class__.__name__]['y_train'] = y_train
            self.metrics[dataset.__class__.__name__]['results'].append(test_scores)
            self.metrics[dataset.__class__.__name__]['names'].append(estimator.name)
        else:
            if not self.initial_metrics:
                self.initial_metrics = {}
            if dataset.__class__.__name__ not in self.initial_metrics.keys():
                self.initial_metrics[dataset.__class__.__name__] = {
                    'train_mean': {},
                    'train_std': {},
                    'test_mean': {},
                    'test_std': {},
                    'fit_times_mean': {},
                    'fit_times_std': {},
                    'train_sizes': {},
                    'X_test': {},
                    'y_test': {},
                    'results': [],
                    'names': [],
                    'validations': []
                }

            self.initial_metrics[dataset.__class__.__name__]['train_mean'][estimator.name] = np.mean(train_scores,
                                                                                                     axis=1)
            self.initial_metrics[dataset.__class__.__name__]['train_std'][estimator.name] = np.std(train_scores, axis=1)
            self.initial_metrics[dataset.__class__.__name__]['test_mean'][estimator.name] = np.mean(test_scores, axis=1)
            self.initial_metrics[dataset.__class__.__name__]['test_std'][estimator.name] = np.std(test_scores, axis=1)
            self.initial_metrics[dataset.__class__.__name__]['fit_times_mean'][estimator.name] = np.mean(fit_times,
                                                                                                         axis=1)
            self.initial_metrics[dataset.__class__.__name__]['fit_times_std'][estimator.name] = np.std(fit_times,
                                                                                                       axis=1)
            self.initial_metrics[dataset.__class__.__name__]['train_sizes'] = train_sizes
            self.initial_metrics[dataset.__class__.__name__]['X_test'] = X_test
            self.initial_metrics[dataset.__class__.__name__]['y_test'] = y_test
            self.initial_metrics[dataset.__class__.__name__]['X_train'] = X_train
            self.initial_metrics[dataset.__class__.__name__]['y_train'] = y_train
            self.initial_metrics[dataset.__class__.__name__]['results'].append(test_scores)
            self.initial_metrics[dataset.__class__.__name__]['names'].append(estimator.name)

    def latext_start_figure(self):
        with open('out.txt', 'a+') as f:
            f.write("\\begin{figure}[!htbp]\n")

    def latex_end_figure(self, caption, fig):
        with open('out.txt', 'a+') as f:
            f.write("""\\caption{%s}
\\label{fig:%s}
\\end{figure}

\\FloatBarrier\n""" % (caption, fig))

    def latex_subgraph(self, dataset, fig, caption, filename):
        latext_template = """\\begin{subfigure}{.24\\textwidth}
  \\centering
  \\includegraphics[width=.9\\textwidth]{%s}
  \\caption{%s}
  \\label{fig:%s_%s}
\\end{subfigure}\n"""
        with open('out.txt', 'a+') as f:
            f.write(latext_template % (filename, caption.replace("_", " "), fig, dataset))

    def plot_learning_curves(self, control=False, suffix=''):
        metric_items = self.metrics if not control else self.initial_metrics
        default = 'default' if control else 'optimized'
        self.latext_start_figure()
        for dataset, data_metrics in metric_items.items():
            train_sizes = data_metrics.get('train_sizes')
            for key, val in data_metrics.get('train_mean').items():
                plt.figure()
                train_std = data_metrics.get('train_std').get(key)
                test_mean = data_metrics.get('test_mean').get(key)
                test_std = data_metrics.get('test_std').get(key)
                plt.plot(train_sizes, val, color='blue', marker='o', markersize=5, label='training accuracy')
                plt.fill_between(train_sizes, val + train_std, val - train_std, alpha=0.15, color='blue')
                plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5,
                         label='test accuracy')
                plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
                plt.ylim(get_y_limits(val, train_std, test_mean, test_std))
                plt.xlabel("Number of training samples")
                plt.ylabel("Score")
                plt.legend()
                filename = 'learnings_%s_%s_%s' % (dataset, key, default)
                chart_path = 'report/images/%s.png' % (filename)
                plt.savefig(chart_path)
                plt.close()
                print(chart_path)
                self.latex_subgraph(dataset=key, fig='learnings_' + dataset, caption=key + '-' + dataset, filename=filename)

        for dataset, data_metrics in metric_items.items():
            train_sizes = data_metrics.get('train_sizes')
            plt.figure()

            for key, val in data_metrics.get('fit_times_mean').items():
                plt.plot(train_sizes, val, label=key)

            plt.xlabel("Number of training samples")
            plt.ylabel("Training Time")
            plt.title("Scalability of the model")
            plt.legend()
            filename = 'scalability_%s_%s%s' % (dataset, default, suffix)
            chart_path = 'report/images/%s.png' % filename
            plt.savefig(chart_path)
            plt.close()
            print(chart_path)
            self.latex_subgraph(dataset, fig='scalability', caption=dataset + " Scalability", filename=filename)

        for dataset, data_metrics in metric_items.items():
            plt.figure()

            for key, val in data_metrics.get('fit_times_mean').items():
                test_mean = data_metrics.get('test_mean').get(key)
                plt.plot(val, test_mean, label=key)

            plt.xlabel("Training Time")
            plt.ylabel("Score")
            plt.title("Performance of the model")
            plt.legend()
            filename = 'performance_%s_%s%s' % (dataset, default, suffix)
            chart_path = 'report/images/%s.png' % filename
            plt.savefig(chart_path)
            plt.close()
            print(chart_path)
            self.latex_subgraph(dataset, fig='performance', caption=dataset + " Time", filename=filename)

        for dataset, data_metrics in metric_items.items():
            plt.figure()
            train_sizes = data_metrics.get('train_sizes')
            for key, val in data_metrics.get('test_mean').items():
                plt.plot(train_sizes, val, label=key)

            plt.xlabel("Number of training samples")
            plt.ylabel("Score")
            plt.title("Performance of the model")
            plt.legend()
            filename = 'performance_all_%s_%s%s' % (dataset, default, suffix)
            chart_path = 'report/images/%s.png' % filename
            plt.savefig(chart_path)
            plt.close()
            print(chart_path)
            self.latex_subgraph(dataset, fig='performance_all', caption=dataset + " Performance", filename=filename)

        for dataset, data_metrics in metric_items.items():
            # boxplot algorithm comparison
            fig = plt.figure()
            fig.suptitle('Algorithm Comparison')
            ax = fig.add_subplot(111)
            try:
                plt.boxplot(metric_items[dataset]['results'])
                ax.set_xticklabels(metric_items[dataset]['names'])
                filename = '%s_%s' % (dataset, default)
                chart_path = 'report/images/%s.png' % filename
                plt.savefig(chart_path)
                plt.close()
                print(chart_path)
                self.latex_subgraph(dataset=dataset, fig='comparisons', caption=dataset, filename=filename)
            except Exception as e:
                print(e)

        self.latex_end_figure(caption="%s Learning Curves" % (default[0].upper() + default[1:]), fig="learning_curves")

    def plot_validation(self):
        self.latext_start_figure()
        for dataset, metrics in self.initial_metrics.items():
            key = dataset
            for params in metrics.get('validations'):
                plt.figure()

                val_x = params.get('x')
                param_name = params.get('param')

                if param_name == 'hidden_layer_sizes':
                    val_x = [''.join([str(y) for y in list(x)]) for x in val_x]

                train_mean = params.get('train_scores')
                test_mean = params.get('test_scores')
                plt.plot(val_x, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
                plt.plot(val_x, test_mean, color='green', linestyle='--', marker='s', markersize=5,
                         label='test accuracy')
                plt.xlabel(param_name)
                if param_name in LOG_SCALES:
                    plt.xscale('log', basex=LOG_SCALES.get(param_name))
                plt.ylabel("Score")
                plt.legend()
                model = params.get('model')
                filename = 'validations_%s_%s_%s' % (dataset, model, param_name)
                chart_path = 'report/images/%s.png' % filename
                plt.savefig(chart_path)
                plt.close()
                print(chart_path)
                self.latex_subgraph(dataset=model, fig='validations_' + dataset, caption=model + '-' + key,
                                    filename=filename)
        self.latex_end_figure(caption="Validation Curves", fig="validations")

    def process_validations(self, dataset, estimator):
        key = dataset.__class__.__name__
        print("Processing Validation Curves for %s with %s." % (key, estimator.name))

        if not self.initial_metrics:
            self.initial_metrics = {}

        if key not in self.initial_metrics.keys():
            self.initial_metrics[key] = {
                'validations': []
            }

        # evaluate each model in turn
        X_train, X_test, y_train, y_test = dataset.get_data(model=estimator.name)

        for k, v in estimator.params.items():
            print("Processing %s with %s." % (k, estimator.name))
            train_scores, test_scores = validation_curve(estimator=estimator.estimator, X=X_train, y=y_train,
                                                         cv=estimator.cv, param_name=k, param_range=v)
            if k not in self.initial_metrics[key].keys():
                self.initial_metrics[key]['validations'].append({
                    'param': k,
                    'x': v,
                    'train_scores': np.mean(train_scores, axis=1),
                    'test_scores': np.mean(test_scores, axis=1),
                    'train_std': np.std(train_scores, axis=1),
                    'test_std': np.mean(test_scores, axis=1),
                    'model': estimator.name
                })

        print("Done")

    def param_selection(self, dataset, estimator):
        key = dataset.__class__.__name__
        print("Searching best params for %s with %s." % (key, estimator.name))
        X_train, X_test, y_train, y_test = dataset.get_data(model=estimator.name)
        grid_search = RandomizedSearchCV(estimator.estimator, param_distributions=estimator.params, scoring='accuracy',
                                         cv=estimator.cv)
        grid_search.fit(X_train, y_train)
        print(grid_search.best_params_)
        self.best_models[key + "_" + estimator.name] = grid_search.best_estimator_
        self.best_params[key + "_" + estimator.name] = grid_search.best_params_

    def print_best_params(self):
        with open('out.txt', 'a+') as f:
            f.write("\n\\section{Best Parameters}\n\n")
            for k, v in self.best_params.items():
                val = "\n%s %s\\\\" % (k, str(v))
                f.write(val.replace('_', ' '))
            f.write("\n\n")

    def get_default_model(self, dataset, estimator):
        dataset_name = dataset.__class__.__name__

        if estimator.name == 'MLP':
            if dataset_name == 'Diabetes':
                estimator.estimator = MLPClassifier(solver='lbfgs')
                return estimator

        return estimator

    def plot_class_distribution(self, dataset):
        plt.figure()
        self.latext_start_figure()
        dataset_name = dataset.__class__.__name__
        X_train, X_test, y_train, y_test = dataset.get_data()
        unique, counts = np.unique(y_train, return_counts=True)

        fig, ax = plt.subplots()
        ax.bar(unique, counts)

        ax.set_xticks(unique)
        ax.set_xticklabels(unique)
        ax.set_xlabel('class')
        ax.set_ylabel('# instances')

        fig.tight_layout()
        filename = 'distribution_%s' % (dataset_name)
        chart_path = 'report/images/%s.png' % filename
        plt.savefig(chart_path)
        plt.close()
        print(chart_path)
        self.latex_subgraph(dataset=dataset_name, fig='balance_' + dataset_name, caption=dataset_name,
                            filename=filename)
        self.latex_end_figure(caption="%s Class Distribution" % (dataset_name[0].upper() + dataset_name[1:]),
                              fig="balance_curve")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find X Coding Quiz')

    parser.add_argument('-m', '--mode', help='Mode', default='debug', dest='mode')
    args = parser.parse_args()

    fast_split = 100
    slow_split = 5
    max_iter = 20000

    if args.mode == 'debu':
        fast_split = 1
        slow_split = 1
        max_iter = 1

    with open('out.txt', 'w') as f:
        f.write("% Start\n")

    kfold_dt = KFold(n_splits=2, shuffle=True, random_state=1)
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)

    processor = Processor()
    datasets = [
        Adult(),
        # Wine(),
        # Credit(),
        Diabetes()
    ]

    for dataset in datasets:
        processor.plot_class_distribution(dataset=dataset)

    estimators = [
        Config(name='DT', estimator=DecisionTreeClassifier(), cv=kfold, params={
            'max_depth': [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
            'min_samples_leaf': [2, 4, 8, 16, 32, 64, 128, 256],
            'min_samples_split': [2, 4, 8, 16, 32, 64, 128, 256],
            'max_leaf_nodes': [2, 4, 8, 16, 32, 64, 128, 256],
            'max_features': [1, 2, 3, 4, 5, 6, 7, 8],
        }),
        Config(name='Boosting', estimator=AdaBoostClassifier(), cv=kfold, params={
            'n_estimators': [1, 2, 4, 8, 16, 32, 64, 128, 256],
        }),
        Config(name='KNN', estimator=KNeighborsClassifier(), cv=kfold, params={
            'n_neighbors': [2, 4, 8, 16, 32]
        }),
        Config(name='SVC_RBF', estimator=SVC(kernel='rbf', cache_size=1000), cv=kfold, params={
            'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
            'max_iter': [200, 400, 800, 1600, 3200, 6400]
        }),
        Config(name='SVC_Linear', estimator=LinearSVC(), cv=kfold, params={
            'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
            'max_iter': [200, 400, 800, 1600, 3200, 6400],
        }),
        Config(name='MLP', estimator=MLPClassifier(), cv=kfold_dt, params={
            'hidden_layer_sizes': get_multi_layers(),
            'max_iter': [200, 400, 800, 1600]
        })
    ]
    for dataset in datasets:
        for estimator in estimators:
            estimator = processor.get_default_model(dataset=dataset, estimator=estimator)
            processor.process(dataset=dataset, estimator=estimator, control=True)
    processor.plot_learning_curves(control=True)

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
