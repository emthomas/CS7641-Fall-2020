import itertools
import json
import random

import mlrose_hiive as mlrose

import argparse
import random as r
import matplotlib.pyplot as plt

import numpy as np
import os
import time

from sklearn.model_selection import KFold
from hw1.main import Processor as P1
from hw1.utils import Adult
from hw1.utils import Diabetes
from hw1.utils import Config

r.seed(1)
np.random.seed(31415)


def acronyms(input):
    if input == 'Mimic':
        return input.upper()
    out = [char for char in input if char.isupper()]
    return ''.join(out)


def title(input):
    out = [x[0].upper() + x[1:] for x in input.split('_')]
    return ''.join(out)


def get_coords(n):
    a_list = [x for x in range(2*n)]
    permutations_object = itertools.combinations(a_list, 3)
    permutations_list = list(permutations_object)
    # return random.choices(permutations_list, k=n)
    return permutations_list


class Processor(object):
    def __init__(self):
        self.iterations = {}
        self.scores = {}
        self.training_times = {}
        self.max_iters = {}
        self.real_iters = {}
        self.real_iters_scores = {}
        self.best_params = {}

    def get_best_param(self, problem, algo, param):
        key = title(algo)
        return self.best_params[problem][key][param].get('val')

    def track_best_params(self, problem, algo, param, value, score):
        key = title(algo)
        if problem not in self.best_params.keys():
            self.best_params[problem] = {}

        if key not in self.best_params[problem].keys():
            self.best_params[problem][key] = {}

        if param not in self.best_params[problem][key].keys():
            self.best_params[problem][key][param] = {'val': 0, 'score': 0}

        if score > self.best_params[problem][key][param].get('score'):
            self.best_params[problem][key][param] = {'val': value, 'score': score}

    def track(self, problem, algo, i, score, training_time=None, max_iter=None, real_iters=None,
              real_iters_scores=None):
        key = title(algo)
        if problem not in self.scores.keys():
            self.scores[problem] = {}
            self.iterations[problem] = {}
            self.training_times[problem] = {}
            self.max_iters[problem] = {}
            self.real_iters[problem] = {}
            self.real_iters_scores[problem] = {}

        if key not in self.scores[problem].keys():
            self.scores[problem][key] = []
            self.iterations[problem][key] = []
            self.training_times[problem][key] = []
            self.max_iters[problem][key] = []
            self.real_iters[problem][key] = []
            self.real_iters_scores[problem][key] = []

        self.scores[problem][key].append(score)
        self.iterations[problem][key].append(i)
        self.training_times[problem][key].append(training_time)
        self.max_iters[problem][key].append(max_iter)
        self.real_iters[problem][key].append(max_iter)
        self.real_iters_scores[problem][key].append(max_iter)

    def latext_start_figure(self):
        with open('out.txt', 'a+') as f:
            f.write("\\begin{figure}[!htbp]\n")

    def latex_end_figure(self, caption, fig):
        with open('out.txt', 'a+') as f:
            f.write("""\\caption{%s}
\\label{fig:%s}
\\end{figure}

\\FloatBarrier\n""" % (caption, fig))

    def latex_subgraph(self, caption, filename):
        latext_template = """\\begin{subfigure}{.3\\textwidth}
  \\centering
  \\includegraphics[width=.9\\textwidth]{%s}
  \\caption{%s}
  \\label{fig:%s}
\\end{subfigure}\n"""
        with open('out.txt', 'a+') as f:
            f.write(latext_template % (filename, caption.replace("_", " "), filename))

    def run_FourPeaks(self, mode=None):
        fitness_fn = mlrose.FourPeaks(t_pct=0.15)
        self.run_complexity(fitness_fn, mode)

    def run_ContinuousPeaks(self, mode=None):
        fitness_fn = mlrose.ContinuousPeaks(t_pct=0.15)
        self.run_complexity(fitness_fn, mode)

    def run_SixPeaks(self, mode=None):
        fitness_fn = mlrose.SixPeaks(t_pct=0.15)
        self.run_complexity(fitness_fn, mode)

    def run_FlipFlop(self, mode=None):
        fitness_fn = mlrose.FlipFlop()
        self.run_complexity(fitness_fn, mode)

    def run_OneMax(self, mode=None):
        fitness_fn = mlrose.OneMax()
        self.run_complexity(fitness_fn, mode)

    def run_MaxKColor(self, mode=None):
        edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4)]
        fitness_fn = mlrose.MaxKColor(edges=edges)
        self.run_complexity(fitness_fn, mode)

    def run_TSP(self, mode=None):
        edges = get_coords(10)
        fitness_fn = mlrose.TravellingSales(distances=edges)
        self.run_complexity(fitness_fn, mode)

    def run_Knap(self, mode=None):
        weights = [10, 5, 2, 8, 15]
        values = [1, 2, 3, 4, 5]
        max_weight_pct = 0.6
        fitness_fn = mlrose.Knapsack(weights, values, max_weight_pct)
        self.run_complexity(fitness_fn, mode)

    def run_complexity(self, fitness_fn, mode=None):
        if mode == 1:
            self.run_ga_hyper_params(fitness_fn)
        elif mode == 2:
            self.run_rhc_hyper_params(fitness_fn)
        elif mode == 3:
            self.run_sa_hyper_params(fitness_fn)
        elif mode == 4:
            self.run_mimic_hyper_params(fitness_fn)
        elif not mode:
            fitness_name = fitness_fn.__class__.__name__
            print("Running %s" % fitness_name)
            init_states = {}
            knap_fitnesses = {}
            tsp_fitnesses = {}
            tries = 1
            for x in 2 ** np.arange(3, 9):
                n = int(x)
                fitness_dists = mlrose.TravellingSales(distances=get_coords(n))
                tsp_fitnesses[n] = fitness_dists
                edges = []
                for x in range(int(n * 0.75)):
                    a = r.randint(0, n - 1)
                    b = r.randint(0, n - 1)
                    while b == a:
                        b = r.randint(0, n - 1)
                    edges.append((a, b))

                fitness_fn_knap = mlrose.MaxKColor(edges=edges)
                init_states[n] = []
                knap_fitnesses[n] = fitness_fn_knap
                for y in range(tries):
                    init_states[n].append(get_init_state(n))

            for n, init_states_list in init_states.items():
                if fitness_name == 'MaxKColor':
                    fitness_fn = knap_fitnesses[n]
                if fitness_name == 'TravellingSales':
                    fitness_fn = tsp_fitnesses[n]
                print(n)
                print('%s: i=%d' % ('random_hill_climb', n))
                total_score = 0
                total_iter = 0
                start = time.time()
                for init_state in init_states_list:
                    problem = mlrose.DiscreteOpt(length=len(init_state), fitness_fn=fitness_fn, maximize=True)
                    if fitness_name == 'TravellingSales':
                        problem = mlrose.TSPOpt(length=n, fitness_fn=fitness_fn)
                    max_attempts = self.get_best_param(problem=fitness_name, algo='random_hill_climb',
                                                       param='max_attempts')
                    restarts = self.get_best_param(problem=fitness_name, algo='random_hill_climb', param='restarts')
                    best_state, best_fitness, curve = mlrose.random_hill_climb(problem,
                                                                               max_attempts=max_attempts,
                                                                               max_iters=10000, random_state=1,
                                                                               curve=True,
                                                                               restarts=restarts)
                    total_iter += len(curve)
                    total_score += np.mean(curve)
                end = time.time()
                print('The fitness at the best state is: ', total_score / tries)
                self.track(problem=fitness_name, algo='random_hill_climb', i=n, score=total_score / tries,
                           training_time=(end - start) / tries, max_iter=total_iter / tries)

            for n, init_states_list in init_states.items():
                if fitness_name == 'MaxKColor':
                    fitness_fn = knap_fitnesses[n]
                if fitness_name == 'TravellingSales':
                    fitness_fn = tsp_fitnesses[n]
                print(n)
                print('%s: i=%d' % ('simulated_annealing', n))
                total_score = 0
                total_iter = 0
                start = time.time()
                for init_state in init_states_list:
                    problem = mlrose.DiscreteOpt(length=len(init_state), fitness_fn=fitness_fn, maximize=True)
                    if fitness_name == 'TravellingSales':
                        problem = mlrose.TSPOpt(length=n, fitness_fn=fitness_fn)
                    max_attempts = self.get_best_param(problem=fitness_name, algo='simulated_annealing',
                                                       param='max_attempts')
                    best_state, best_fitness, curve = mlrose.simulated_annealing(problem, max_attempts=max_attempts,
                                                                                 max_iters=10000, random_state=1,
                                                                                 curve=True)
                    total_score += np.mean(curve)
                    total_iter += len(curve)
                end = time.time()
                print('The fitness at the best state is: ', total_score / tries)
                self.track(problem=fitness_name, algo='simulated_annealing', i=n, score=total_score / tries,
                           training_time=(end - start) / tries, max_iter=total_iter / tries)

            for n, init_states_list in init_states.items():
                if fitness_name == 'MaxKColor':
                    fitness_fn = knap_fitnesses[n]
                if fitness_name == 'TravellingSales':
                    fitness_fn = tsp_fitnesses[n]
                print(n)
                print('%s: i=%d' % ('genetic_alg', n))
                total_score = 0
                total_iter = 0
                start = time.time()
                for init_state in init_states_list:
                    problem = mlrose.DiscreteOpt(length=len(init_state), fitness_fn=fitness_fn, maximize=True)
                    if fitness_name == 'TravellingSales':
                        problem = mlrose.TSPOpt(length=n, fitness_fn=fitness_fn)
                    mutation_prob = self.get_best_param(problem=fitness_name, algo='genetic_alg', param='mutation_prob')
                    pop_size = self.get_best_param(problem=fitness_name, algo='genetic_alg', param='pop_size')
                    best_state, best_fitness, curve = mlrose.genetic_alg(problem, pop_size=pop_size,
                                                                         mutation_prob=mutation_prob,
                                                                         max_iters=10000, random_state=1, curve=True)
                    total_score += np.mean(curve)
                    total_iter += len(curve)
                end = time.time()
                print('The fitness at the best state is: ', total_score / tries)
                self.track(problem=fitness_name, algo='genetic_alg', i=n, score=total_score / tries,
                           training_time=(end - start) / tries, max_iter=total_iter / tries)

            for n, init_states_list in init_states.items():
                if fitness_name == 'MaxKColor':
                    fitness_fn = knap_fitnesses[n]
                if fitness_name == 'TravellingSales':
                    fitness_fn = tsp_fitnesses[n]
                print('%s: i=%d' % ('mimic', n))
                if n > 256:
                    break
                total_score = 0
                total_iter = 0
                start = time.time()
                for init_state in init_states_list:
                    problem = mlrose.DiscreteOpt(length=len(init_state), fitness_fn=fitness_fn, maximize=True)
                    if fitness_name == 'TravellingSales':
                        problem = mlrose.TSPOpt(length=n, fitness_fn=fitness_fn)
                    keep_pct = self.get_best_param(problem=fitness_name, algo='mimic', param='keep_pct')
                    pop_size = self.get_best_param(problem=fitness_name, algo='mimic', param='pop_size')
                    best_state, best_fitness, curve = mlrose.mimic(problem, max_iters=10000, random_state=1, curve=True,
                                                                   pop_size=pop_size, keep_pct=keep_pct,
                                                                   max_attempts=10)
                    total_score += np.mean(curve)
                    total_iter += len(curve)
                end = time.time()
                print('The fitness at the best state is: ', total_score / tries)
                self.track(problem=fitness_name, algo='mimic', i=n, score=total_score / tries,
                           training_time=(end - start) / tries, max_iter=total_iter / tries)

    def plot_toys_curves(self, is_log=True):
        self.latext_start_figure()
        for i, j in self.iterations.items():
            plt.figure()
            for a, b in j.items():
                score = self.scores.get(i).get(a)
                plt.plot(b, score, label=a)

            plt.xlabel("Problem Size")
            if is_log:
                plt.xscale('log', basex=2)
            plt.ylabel("Score")
            plt.title("Performance of the model")
            plt.legend()
            filename = '%s_scores_complexity' % (i)
            chart_path = 'report/images/%s.png' % filename
            plt.savefig(chart_path)
            plt.close()
            print(chart_path)
            self.latex_subgraph(caption=i + " Scores", filename=filename)

        for i, j in self.iterations.items():
            plt.figure()
            for a, b in j.items():
                score = self.training_times.get(i).get(a)
                plt.plot(b, score, label=a)

            plt.xlabel("Problem Size")
            if is_log:
                plt.xscale('log', basex=2)
            plt.ylabel("Training Time")
            plt.title("Performance of the model")
            plt.legend()
            filename = '%s_times_complexity' % (i)
            chart_path = 'report/images/%s.png' % filename
            plt.savefig(chart_path)
            plt.close()
            print(chart_path)
            self.latex_subgraph(caption=i + " Times", filename=filename)

        for i, j in self.iterations.items():
            plt.figure()
            for a, b in j.items():
                score = self.max_iters.get(i).get(a)
                plt.plot(b, score, label=a)

            plt.xlabel("Problem Size")
            if is_log:
                plt.xscale('log', basex=2)
            plt.ylabel("Max Iteration")
            plt.title("Iterations to convergence")
            plt.legend()
            filename = '%s_iters_to_convergence_complexity' % (i)
            chart_path = 'report/images/%s.png' % filename
            plt.savefig(chart_path)
            plt.close()
            print(chart_path)
            self.latex_subgraph(caption=i + " Times", filename=filename)

        self.latex_end_figure(caption='Plots', fig='plots')

    def run_mimic_hyper_params(self, fitness_fn):
        fitness_name = fitness_fn.__class__.__name__
        print("Running %s" % fitness_name)
        init_states = {}
        knap_fitnesses = {}
        tsp_fitnesses = {}
        tries = 1
        for x in 2 ** np.arange(6, 7):
            n = int(x)
            fitness_dists = mlrose.TravellingSales(distances=get_coords(n))
            tsp_fitnesses[n] = fitness_dists
            edges = []
            for x in range(int(n * 0.75)):
                a = r.randint(0, n - 1)
                b = r.randint(0, n - 1)
                while b == a:
                    b = r.randint(0, n - 1)
                edges.append((a, b))

            fitness_fn_knap = mlrose.MaxKColor(edges=edges)
            init_states[n] = []
            knap_fitnesses[n] = fitness_fn_knap
            for y in range(tries):
                init_states[n].append(get_init_state(n))

        for n, init_states_list in init_states.items():
            if fitness_name == 'MaxKColor':
                    fitness_fn = knap_fitnesses[n]
            if fitness_name == 'TravellingSales':
                fitness_fn = tsp_fitnesses[n]
            print(n)
            print('%s: i=%d' % ('mimic', n))

            for init_state in init_states_list:
                problem = mlrose.DiscreteOpt(length=len(init_state), fitness_fn=fitness_fn, maximize=True)
                if fitness_name == 'TravellingSales':
                    problem = mlrose.TSPOpt(length=n, fitness_fn=fitness_fn)
                for pop_size in range(100, 1000, 100):
                    total_score = 0
                    total_iter = 0
                    best_state, best_fitness, curve = mlrose.mimic(problem, pop_size=pop_size, keep_pct=0.1,
                                                                   max_iters=10000, random_state=1, curve=True,
                                                                   max_attempts=10)
                    total_score += np.max(curve)
                    total_iter += len(curve)
                    print('The fitness at the best state is: ', total_score / tries, '. Pop size: ', pop_size)
                    self.track_best_params(problem=fitness_name, algo='mimic', param='pop_size', score=total_score,
                                           value=pop_size)

                pop_size = self.get_best_param(problem=fitness_name, algo='mimic', param='pop_size')
                for keep_pct in range(0, 10, 1):
                    total_score = 0
                    total_iter = 0
                    best_state, best_fitness, curve = mlrose.mimic(problem, pop_size=pop_size,
                                                                   keep_pct=1.0 * keep_pct / 10,
                                                                   max_iters=10000, random_state=1, curve=True,
                                                                   max_attempts=10)
                    total_score += np.max(curve)
                    total_iter += len(curve)
                    print('The fitness at the best state is: ', total_score / tries, '. keep_pct: ',
                          1.0 * keep_pct / 10)
                    self.track_best_params(problem=fitness_name, algo='mimic', param='keep_pct',
                                           score=total_score, value=1.0 * keep_pct / 10)

    def run_ga_hyper_params(self, fitness_fn):
        fitness_name = fitness_fn.__class__.__name__
        print("Running %s" % fitness_name)
        init_states = {}
        knap_fitnesses = {}
        tsp_fitnesses = {}
        tries = 1
        for x in 2 ** np.arange(6, 7):
            n = int(x)
            fitness_dists = mlrose.TravellingSales(distances=get_coords(n))
            tsp_fitnesses[n] = fitness_dists
            edges = []
            for x in range(int(n * 0.75)):
                a = r.randint(0, n - 1)
                b = r.randint(0, n - 1)
                while b == a:
                    b = r.randint(0, n - 1)
                edges.append((a, b))

            fitness_fn_knap = mlrose.MaxKColor(edges=edges)
            init_states[n] = []
            knap_fitnesses[n] = fitness_fn_knap
            for y in range(tries):
                init_states[n].append(get_init_state(n))

        for n, init_states_list in init_states.items():
            if fitness_name == 'MaxKColor':
                    fitness_fn = knap_fitnesses[n]
            if fitness_name == 'TravellingSales':
                fitness_fn = tsp_fitnesses[n]
            print(n)
            print('%s: i=%d' % ('genetic_alg', n))

            for init_state in init_states_list:
                problem = mlrose.DiscreteOpt(length=len(init_state), fitness_fn=fitness_fn, maximize=True)
                if fitness_name == 'TravellingSales':
                    problem = mlrose.TSPOpt(length=n, fitness_fn=fitness_fn)
                for pop_size in range(100, 1000, 100):
                    total_score = 0
                    total_iter = 0
                    best_state, best_fitness, curve = mlrose.genetic_alg(problem, pop_size=pop_size, mutation_prob=0.1,
                                                                         max_iters=10000, random_state=1, curve=True)
                    total_score += np.max(curve)
                    total_iter += len(curve)
                    print('The fitness at the best state is: ', total_score / tries, '. Pop size: ', pop_size)
                    self.track_best_params(problem=fitness_name, algo='genetic_alg', param='pop_size',
                                           score=total_score, value=pop_size)

                pop_size = self.get_best_param(problem=fitness_name, algo='genetic_alg', param='pop_size')
                for mutation_prob in range(0, 11, 1):
                    total_score = 0
                    total_iter = 0
                    best_state, best_fitness, curve = mlrose.genetic_alg(problem, pop_size=pop_size,
                                                                         mutation_prob=mutation_prob / 10,
                                                                         max_iters=10000, random_state=1, curve=True)
                    total_score += np.max(curve)
                    total_iter += len(curve)
                    print('The fitness at the best state is: ', total_score / tries, '. Rate: ', mutation_prob / 10)
                    self.track_best_params(problem=fitness_name, algo='genetic_alg', param='mutation_prob',
                                           score=total_score, value=mutation_prob / 10)

    def run_rhc_hyper_params(self, fitness_fn):
        fitness_name = fitness_fn.__class__.__name__
        print("Running %s" % fitness_name)
        init_states = {}
        knap_fitnesses = {}
        tsp_fitnesses = {}
        tries = 1
        for x in 2 ** np.arange(6, 7):
            n = int(x)
            fitness_dists = mlrose.TravellingSales(distances=get_coords(n))
            tsp_fitnesses[n] = fitness_dists
            edges = []
            for x in range(int(n * 0.75)):
                a = r.randint(0, n - 1)
                b = r.randint(0, n - 1)
                while b == a:
                    b = r.randint(0, n - 1)
                edges.append((a, b))

            fitness_fn_knap = mlrose.MaxKColor(edges=edges)
            init_states[n] = []
            knap_fitnesses[n] = fitness_fn_knap
            for y in range(tries):
                init_states[n].append(get_init_state(n))

        for n, init_states_list in init_states.items():
            if fitness_name == 'MaxKColor':
                    fitness_fn = knap_fitnesses[n]
            if fitness_name == 'TravellingSales':
                fitness_fn = tsp_fitnesses[n]
            print(n)
            print('%s: i=%d' % ('random_hill_climb', n))

            for init_state in init_states_list:
                problem = mlrose.DiscreteOpt(length=len(init_state), fitness_fn=fitness_fn, maximize=True)
                if fitness_name == 'TravellingSales':
                    problem = mlrose.TSPOpt(length=n, fitness_fn=fitness_fn)
                for max_attempts in range(10, 110, 10):
                    total_score = 0
                    total_iter = 0
                    best_state, best_fitness, curve = mlrose.random_hill_climb(problem,
                                                                               max_attempts=max_attempts,
                                                                               max_iters=10000, random_state=1,
                                                                               curve=True,
                                                                               restarts=10)
                    total_score += np.max(curve)
                    total_iter += len(curve)
                    print('The fitness at the best state is: ', total_score / tries, '. Max Attempts: ', max_attempts)
                    self.track_best_params(problem=fitness_name, algo='random_hill_climb', param='max_attempts',
                                           score=total_score, value=max_attempts)

                max_attempts = self.get_best_param(problem=fitness_name, algo='random_hill_climb', param='max_attempts')
                for restarts in range(10, 110, 10):
                    total_score = 0
                    total_iter = 0
                    best_state, best_fitness, curve = mlrose.random_hill_climb(problem,
                                                                               max_attempts=max_attempts,
                                                                               max_iters=10000, random_state=1,
                                                                               curve=True,
                                                                               restarts=restarts)
                    total_score += np.max(curve)
                    total_iter += len(curve)
                    print('The fitness at the best state is: ', total_score / tries, '. restarts: ', restarts)
                    self.track_best_params(problem=fitness_name, algo='random_hill_climb', param='restarts',
                                           score=total_score, value=restarts)

    def run_sa_hyper_params(self, fitness_fn):
        fitness_name = fitness_fn.__class__.__name__
        print("Running %s" % fitness_name)
        init_states = {}
        knap_fitnesses = {}
        tsp_fitnesses = {}
        tries = 1
        for x in 2 ** np.arange(6, 7):
            n = int(x)
            fitness_dists = mlrose.TravellingSales(distances=get_coords(n))
            tsp_fitnesses[n] = fitness_dists
            edges = []
            for x in range(int(n * 0.75)):
                a = r.randint(0, n - 1)
                b = r.randint(0, n - 1)
                while b == a:
                    b = r.randint(0, n - 1)
                edges.append((a, b))

            fitness_fn_knap = mlrose.MaxKColor(edges=edges)
            init_states[n] = []
            knap_fitnesses[n] = fitness_fn_knap
            for y in range(tries):
                init_states[n].append(get_init_state(n))

        for n, init_states_list in init_states.items():
            if fitness_name == 'MaxKColor':
                    fitness_fn = knap_fitnesses[n]
            if fitness_name == 'TravellingSales':
                fitness_fn = tsp_fitnesses[n]
            print(n)
            print('%s: i=%d' % ('simulated_annealing', n))

            for init_state in init_states_list:
                problem = mlrose.DiscreteOpt(length=len(init_state), fitness_fn=fitness_fn, maximize=True)
                if fitness_name == 'TravellingSales':
                    problem = mlrose.TSPOpt(length=n, fitness_fn=fitness_fn)
                for max_attempts in range(10, 110, 10):
                    total_score = 0
                    total_iter = 0
                    best_state, best_fitness, curve = mlrose.simulated_annealing(problem, max_attempts=max_attempts,
                                                                                 max_iters=10000, random_state=1,
                                                                                 curve=True)
                    total_score += np.max(curve)
                    total_iter += len(curve)
                    print('The fitness at the best state is: ', total_score / tries, '. Max Attempts: ', max_attempts)
                    self.track_best_params(problem=fitness_name, algo='simulated_annealing', param='max_attempts',
                                           score=total_score, value=max_attempts)

    def load_best_params(self):
        with open('best_params.json') as json_file:
            self.best_params = json.load(json_file)

    def print_best_params_table(self):
        tup = []
        for a, b in self.best_params.items():
            for x, y in b.items():
                for e, f in y.items():
                    tup.append((a, acronyms(x), title(e), f.get('val')))
        tup.sort(key=lambda x: (x[1], x[2]))
        with open('best_params.txt', 'w+') as f:
            for t in tup:
                f.write(' & '.join([str(x) for x in t]) + " \\\\" + "\n")


def get_init_state(n):
    coords = []
    for i in range(n):
        coords.append(0)
    return coords


def run_toy_problems():
    processor = Processor()
    processor.run_FlipFlop(mode=1)
    processor.run_ContinuousPeaks(mode=1)
    processor.run_MaxKColor(mode=1)
    processor.run_FlipFlop(mode=2)
    processor.run_ContinuousPeaks(mode=2)
    processor.run_MaxKColor(mode=2)
    processor.run_FlipFlop(mode=3)
    processor.run_ContinuousPeaks(mode=3)
    processor.run_MaxKColor(mode=3)
    processor.run_FlipFlop(mode=4)
    processor.run_ContinuousPeaks(mode=4)
    processor.run_MaxKColor(mode=4)
    # processor.run_TSP(mode=1)
    # processor.run_TSP(mode=2)
    # processor.run_TSP(mode=3)
    # processor.run_TSP(mode=4)
    with open('best_params.json', 'w+') as f:
        f.write(json.dumps(processor.best_params, indent=2) + "\n")
    print(json.dumps(processor.best_params, indent=2))

    processor.load_best_params()
    processor.print_best_params_table()

    processor.run_FlipFlop()
    processor.plot_toys_curves(is_log=True)
    processor.run_ContinuousPeaks()
    processor.plot_toys_curves(is_log=True)
    processor.run_MaxKColor()
    processor.plot_toys_curves(is_log=True)
    # processor.run_TSP()
    # processor.plot_toys_curves(is_log=True)


def run_nn_opt():
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    processor = P1()
    datasets = [Diabetes()]
    # 'random_hill_climb', 'simulated_annealing', 'genetic_alg'
    estimators = [
        Config(name='NN_%s' % title('random_hill_climb'),
               estimator=mlrose.NeuralNetwork(
                   algorithm='random_hill_climb',
                   random_state=1,
                   max_iters=200,
                   hidden_nodes=[64],
                   early_stopping=True,
               ),
               cv=kfold,
               params={
                   'restarts': [0, 10, 20, 30, 40, 50]
               }),
        Config(name='NN_%s' % title('simulated_annealing'),
               estimator=mlrose.NeuralNetwork(
                   algorithm='simulated_annealing',
                   random_state=1,
                   max_iters=200,
                   hidden_nodes=[64],
                   early_stopping=True,
               ),
               cv=kfold,
               params={
                   'max_iters': [200]
               }),
        Config(name='NN_%s' % title('genetic_alg'),
               estimator=mlrose.NeuralNetwork(
                   algorithm='genetic_alg',
                   random_state=1,
                   max_iters=200,
                   hidden_nodes=[64],
                   early_stopping=True,
               ),
               cv=kfold,
               params={
                   'pop_size': [100, 200, 300, 400, 500, 600, 700, 800, 900],
                   'mutation_prob': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
               }),
    ]

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


if __name__ == "__main__":
    try:
        os.makedirs('report/images', exist_ok=True)
        print("Directory created successfully.")
    except OSError as error:
        print("Directory '%s' can not be created")

    parser = argparse.ArgumentParser(description='Find X Coding Quiz')

    parser.add_argument('-m', '--mode', help='Mode', default='debug', dest='mode')
    args = parser.parse_args()

    run_toy_problems()
    run_nn_opt()
