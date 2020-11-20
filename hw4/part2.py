import os
import re
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym.envs.toy_text.frozen_lake import generate_random_map

from hw1.main import Processor


def Taxi_Experiments():
    environment = 'Taxi-v1'
    env = gym.make(environment)
    env = env.unwrapped
    desc = env.unwrapped.desc
    gammas = [(x + 0.5) / 10 for x in range(3, 10)]

    policy_time_array = []
    policy_iters = []
    policy_list_scores = []

    ### POLICY ITERATION TAXI: ####
    print('POLICY ITERATION WITH TAXI')
    for gamma in gammas:
        all_iters = []
        all_times = []
        all_scores = []
        for i in range(1):
            env.reset()
            st = time.time()
            best_policy, k = policy_iteration(env, gamma=gamma)
            scores = evaluate_policy(env, best_policy, gamma=gamma)
            end = time.time()
            all_iters.append(k)
            all_times.append(end - st)
            all_scores.append(np.mean(scores))
        policy_list_scores.append(np.mean(all_scores))
        policy_iters.append(np.mean(all_iters))
        policy_time_array.append(np.mean(all_times))

    value_time_array = []
    value_iters = []
    value_list_scores = []
    #### VALUE ITERATION TAXI: #####
    print('VALUE ITERATION WITH TAXI')
    for gamma in gammas:
        all_iters = []
        all_times = []
        all_scores = []
        for i in range(1):
            env.reset()
            st = time.time()
            best_value, k = value_iteration(env, gamma=gamma)
            best_policy = extract_policy(env, best_value, gamma=gamma)
            scores = evaluate_policy(env, best_policy, gamma=gamma)
            end = time.time()
            all_iters.append(k)
            all_times.append(end - st)
            all_scores.append(np.mean(scores))
        value_list_scores.append(np.mean(all_scores))
        value_iters.append(np.mean(all_iters))
        value_time_array.append(np.mean(all_times))

    plt.figure()
    plt.plot(gammas, policy_time_array, label="Policy Iteration")
    plt.plot(gammas, value_time_array, label="Value Iteration")
    plt.xlabel('Discount Factor')
    plt.title('Taxi - Execution Time Analysis')
    plt.ylabel('Execution Time (s)')
    plt.legend(loc='best')

    dataset_name = 'Taxi-v1'
    processor = Processor()
    title = "Taxi - Execution Time Analysis"
    plt.title("%s" % title)
    filename = '%s_%s_%s' % ('times', 'pi_vi', dataset_name)
    chart_path = 'report/images/%s.png' % filename
    plt.savefig(chart_path)
    processor.latext_start_figure()
    processor.latex_subgraph(dataset=dataset_name, fig=filename, caption='Times', filename=filename)

    plt.figure()
    plt.plot(gammas, policy_list_scores, label="Policy Iteration")
    plt.plot(gammas, value_list_scores, label="Value Iteration")
    plt.xlabel('Discount Factor')
    plt.ylabel('Average Rewards')
    plt.title('Taxi - Reward Analysis')
    plt.legend(loc='best')
    title = "Taxi - Reward Analysis"
    plt.title("%s" % title)
    filename = '%s_%s_%s' % ('rewards', 'pi_vi', dataset_name)
    chart_path = 'report/images/%s.png' % filename
    plt.savefig(chart_path)
    processor.latext_start_figure()
    processor.latex_subgraph(dataset=dataset_name, fig=filename, caption='Rewards', filename=filename)

    plt.figure()
    plt.plot(gammas, policy_iters, label="Policy Iteration")
    plt.plot(gammas, value_iters, label="Value Iteration")
    plt.xlabel('Discount Factor')
    plt.ylabel('Iterations to Converge')
    plt.title('Taxi - Convergence Analysis')
    plt.legend(loc='best')
    title = "Taxi - Convergence Analysis"
    plt.title("%s" % title)
    filename = '%s_%s_%s' % ('convergence', 'pi_vi', dataset_name)
    chart_path = 'report/images/%s.png' % filename
    plt.savefig(chart_path)
    processor.latext_start_figure()
    processor.latex_subgraph(dataset=dataset_name, fig=filename, caption='Convergence', filename=filename)

    title = "Taxi - PI and VI Charts"
    fig = re.sub('[^0-9a-zA-Z]+', '_', title.lower())
    processor.latex_end_figure(caption=title, fig=fig)


def Frozen_Lake_Experiments(env, environment='FrozenLake-v0'):
    env = env.unwrapped
    gammas = [(x + 0.5) / 10 for x in range(3, 10)]

    policy_time_array = []
    policy_iters = []
    policy_list_scores = []

    ### POLICY ITERATION TAXI: ####
    print('POLICY ITERATION WITH FROZEN LAKE')
    for gamma in gammas:
        print(f"gamma: {gamma}")
        all_iters = []
        all_times = []
        all_scores = []
        for episode in range(1):
            env.reset()
            st = time.time()
            best_policy, k = policy_iteration(env, gamma=gamma)
            scores = evaluate_policy(env, best_policy, gamma=gamma)
            end = time.time()
            all_iters.append(k)
            all_times.append(end - st)
            all_scores.append(np.mean(scores))
        policy_list_scores.append(np.mean(all_scores))
        policy_iters.append(np.mean(all_iters))
        policy_time_array.append(np.mean(all_times))

    value_time_array = []
    value_iters = []
    value_list_scores = []
    #### VALUE ITERATION TAXI: #####
    print('VALUE ITERATION WITH FROZEN LAKE')
    for gamma in gammas:
        all_iters = []
        all_times = []
        all_scores = []
        for episode in range(1):
            env.reset()
            st = time.time()
            best_value, k = value_iteration(env, gamma=gamma)
            best_policy = extract_policy(env, best_value, gamma=gamma)
            scores = evaluate_policy(env, best_policy, gamma=gamma)
            end = time.time()
            all_iters.append(k)
            all_times.append(end - st)
            all_scores.append(np.mean(scores))
        value_list_scores.append(np.mean(all_scores))
        value_iters.append(np.mean(all_iters))
        value_time_array.append(np.mean(all_times))

    plt.figure()
    plt.plot(gammas, policy_time_array, label="Policy Iteration")
    plt.plot(gammas, value_time_array, label="Value Iteration")
    plt.xlabel('Discount Factor')
    plt.title('FrozenLake - Execution Time Analysis')
    plt.ylabel('Execution Time (s)')
    plt.legend(loc='best')

    dataset_name = environment
    processor = Processor()
    title = "FrozenLake - Execution Time Analysis"
    plt.title("%s" % title)
    filename = '%s_%s_%s' % ('times', 'pi_vi', dataset_name)
    chart_path = 'report/images/%s.png' % filename
    plt.savefig(chart_path)
    print(chart_path)
    processor.latext_start_figure()
    processor.latex_subgraph(dataset=dataset_name, fig=filename, caption='Times', filename=filename)

    plt.figure()
    plt.plot(gammas, policy_list_scores, label="Policy Iteration")
    plt.plot(gammas, value_list_scores, label="Value Iteration")
    plt.xlabel('Discount Factor')
    plt.ylabel('Average Rewards')
    plt.title('FrozenLake - Reward Analysis')
    plt.legend(loc='best')
    title = "FrozenLake - Reward Analysis"
    plt.title("%s" % title)
    filename = '%s_%s_%s' % ('rewards', 'pi_vi', dataset_name)
    chart_path = 'report/images/%s.png' % filename
    plt.savefig(chart_path)
    print(chart_path)
    processor.latext_start_figure()
    processor.latex_subgraph(dataset=dataset_name, fig=filename, caption='Rewards', filename=filename)

    plt.figure()
    plt.plot(gammas, policy_iters, label="Policy Iteration")
    plt.plot(gammas, value_iters, label="Value Iteration")
    plt.xlabel('Discount Factor')
    plt.ylabel('Iterations to Converge')
    plt.title('FrozenLake - Convergence Analysis')
    plt.legend(loc='best')
    title = "FrozenLake - Convergence Analysis"
    plt.title("%s" % title)
    filename = '%s_%s_%s' % ('convergence', 'pi_vi', dataset_name)
    chart_path = 'report/images/%s.png' % filename
    plt.savefig(chart_path)
    print(chart_path)
    processor.latext_start_figure()
    processor.latex_subgraph(dataset=dataset_name, fig=filename, caption='Convergence', filename=filename)

    title = "FrozenLake - PI and VI Charts"
    fig = re.sub('[^0-9a-zA-Z]+', '_', title.lower())
    processor.latex_end_figure(caption=title, fig=fig)


def run_episode(env, policy, gamma, render=True):
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done, _ = env.step(int(policy[obs]))
        if done and reward == 0:
            env.reset()
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward


def evaluate_policy(env, policy, gamma, n=100):
    scores = [run_episode(env, policy, gamma, False) for _ in range(n)]
    return np.mean(scores)


def extract_policy(env, v, gamma):
    policy = np.zeros(env.nS)
    for s in range(env.nS):
        q_sa = np.zeros(env.nA)
        for a in range(env.nA):
            q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in env.P[s][a]])
        policy[s] = np.argmax(q_sa)
    return policy


def compute_policy_v(env, policy, gamma):
    v = np.zeros(env.nS)
    eps = 1e-5
    while True:
        prev_v = np.copy(v)
        for s in range(env.nS):
            policy_a = policy[s]
            v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, is_done in env.P[s][policy_a]])
        if (np.sum((np.fabs(prev_v - v))) <= eps):
            break
    return v


def policy_iteration(env, gamma):
    policy = np.random.choice(env.nA, size=(env.nS))
    max_iters = 200000
    desc = env.unwrapped.desc
    for i in range(max_iters):
        old_policy_v = compute_policy_v(env, policy, gamma)
        new_policy = extract_policy(env, old_policy_v, gamma)
        if (np.all(policy == new_policy)):
            k = i + 1
            break
        policy = new_policy
    return policy, k


def value_iteration(env, gamma):
    v = np.zeros(env.nS)  # initialize value-function
    max_iters = 100000
    eps = 1e-20
    desc = env.unwrapped.desc
    for i in range(max_iters):
        prev_v = np.copy(v)
        for s in range(env.nS):
            q_sa = [sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(env.nA)]
            v[s] = max(q_sa)
        if (np.sum(np.fabs(prev_v - v)) <= eps):
            k = i + 1
            break
    return v, k


def plot_policy_map(title, policy, map_desc, color_map, direction_map):
    fig = plt.figure()
    ax = fig.add_subplot(111, xlim=(0, policy.shape[1]), ylim=(0, policy.shape[0]))
    font_size = 'x-large'
    if policy.shape[1] > 16:
        font_size = 'small'
    plt.title(title)
    for i in range(policy.shape[0]):
        for j in range(policy.shape[1]):
            y = policy.shape[0] - i - 1
            x = j
            p = plt.Rectangle([x, y], 1, 1)
            p.set_facecolor(color_map[map_desc[i, j]])
            ax.add_patch(p)

            text = ax.text(x + 0.5, y + 0.5, direction_map[policy[i, j]], weight='bold', size=font_size,
                           horizontalalignment='center', verticalalignment='center', color='w')

    plt.axis('off')
    plt.xlim((0, policy.shape[1]))
    plt.ylim((0, policy.shape[0]))
    plt.tight_layout()
    clean_title = re.sub('[^0-9a-zA-Z]+', '_', title.lower())
    # plt.savefig('report/images/'+clean_title + str('.png'))
    plt.show()
    plt.close()

    return (plt)


def colors_lake():
    return {
        b'S': 'green',
        b'F': 'skyblue',
        b'H': 'black',
        b'G': 'gold',
    }


def directions_lake():
    return {
        3: '⬆',
        2: '➡',
        1: '⬇',
        0: '⬅'
    }


def actions_taxi():
    return {
        0: '⬇',
        1: '⬆',
        2: '➡',
        3: '⬅',
        4: 'P',
        5: 'D'
    }


def colors_taxi():
    return {
        b'+': 'red',
        b'-': 'green',
        b'R': 'yellow',
        b'G': 'blue',
        b'Y': 'gold'
    }


def run_experiments_part1():
    try:
        os.makedirs('report/images', exist_ok=True)
        print("Directory created successfully.")
    except OSError as error:
        print("Directory '%s' can not be created")
    print('STARTING EXPERIMENTS')
    env = gym.make("FrozenLake-v0")
    Frozen_Lake_Experiments(env=env, environment="FrozenLake-v0")

    random_map = generate_random_map(size=40, p=0.8)
    env = gym.make("FrozenLake-v0",  desc=random_map)
    Frozen_Lake_Experiments(env=env, environment="FrozenLake-40x40")

    Taxi_Experiments()
    print('END OF EXPERIMENTS')


if __name__ == '__main__':
    run_experiments_part1()
