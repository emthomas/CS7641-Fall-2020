# https://www.kaggle.com/angps95/intro-to-reinforcement-learning-with-openai-gym
import random
import re
import sys
import time
import warnings

import gym  # openAi gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gym.envs.toy_text.frozen_lake import generate_random_map

from hw1.main import Processor

warnings.filterwarnings('ignore')

random.seed(1)
np.random.seed(31415)


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


def smooth(in_data, window=10):
    mean = pd.Series(in_data).rolling(window, min_periods=window).mean().to_list()
    return mean


class Policy(object):
    def __init__(self,
                 policy,
                 q_table,
                 name,
                 iterations_to_converge,
                 time_to_converge=.0):
        self.policy = policy
        self.q_table = q_table
        self.name = name
        self.iterations_to_converge = iterations_to_converge
        self.time_to_converge = time_to_converge


class Metric(object):
    def __init__(self):
        self.max = -99999
        self.episode = 99999
        self.steps = 0
        self.eps = 0.0

    def track(self, reward, episode=0, step=0, epsilon=0.0):
        if reward == self.max:
            pass
        if reward >= self.max:
            self.max = reward
            self.episode = episode
            self.steps = step
            self.eps = epsilon

    def __str__(self):
        return f"Reward: {self.max}; Episode: {self.episode}; Step: {self.steps}; Epsilon: {self.eps}"


def random_policy_steps_count(env, max_steps=10000):
    state = env.reset()
    counter = 0
    penalties = 0
    reward = None
    while reward != 20 and reward != 1 and counter < max_steps:
        state, reward, done, info = env.step(env.action_space.sample())
        if done:
            env.reset()
            penalties +=1
        counter += 1

    return counter, penalties


def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.observation_space.n is a number of states in the environment. 
            env.action_space.n is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        Vector of length env.observation_space.n representing the value function.
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.observation_space.n)
    while True:
        # TODO: Implement!
        delta = 0  # delta = change in value of state from one iteration to next

        for state in range(env.observation_space.n):  # for all states
            val = 0  # initiate value as 0

            for action, act_prob in enumerate(policy[state]):  # for all actions/action probabilities
                # transition probabilities,state,rewards of each action
                for prob, next_state, reward, done in env.P[state][action]:
                    val += act_prob * prob * (reward + discount_factor * V[next_state])  # eqn to calculate
            delta = max(delta, np.abs(val - V[state]))
            V[state] = val
        if delta < theta:  # break if the change in value is less than the threshold (theta)
            break
    return np.array(V)


def policy_iteration(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.

    Args:
        env: The OpenAI envrionment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.

    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.

    """

    def one_step_lookahead(env, state, V):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.observation_space.n

        Returns:
            A vector of length env.action_space.n containing the expected value of each action.
        """
        A = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A

    # Start with a random policy
    policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n

    i = 0
    st = time.time()
    while True:
        print(f"Policy Iteration #{i}")
        i += 1
        # Implement this!
        curr_pol_val = policy_eval_fn(policy, env, discount_factor)  # eval current policy
        policy_stable = True  # Check if policy did improve (Set it as True first)
        for state in range(env.observation_space.n):  # for each states
            chosen_act = np.argmax(policy[state])  # best action (Highest prob) under current policy
            act_values = one_step_lookahead(env, state, curr_pol_val)  # use one step lookahead to find action values
            best_act = np.argmax(act_values)  # find best action
            if chosen_act != best_act:
                policy_stable = False  # Greedily find best action
            policy[state] = np.eye(env.action_space.n)[best_act]  # update 
        if policy_stable or i > 15:
            return Policy(policy=policy
                          , q_table=curr_pol_val
                          , name="Policy Iteration"
                          , iterations_to_converge=i
                          , time_to_converge=time.time()-st)

    # return Policy(policy=policy, q_table=np.zeros(env.observation_space.n), name="Policy Iteration")


def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.

    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.observation_space.n is a number of states in the environment. 
            env.action_space.n is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.        
    """

    def one_step_lookahead(env, state, V):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.observation_space.n

        Returns:
            A vector of length env.action_space.n containing the expected value of each action.
        """
        A = np.zeros(env.action_space.n)
        for act in range(env.action_space.n):
            for prob, next_state, reward, done in env.P[state][act]:
                A[act] += prob * (reward + discount_factor * V[next_state])
        return A

    V = np.zeros(env.observation_space.n)
    i = 0
    st = time.time()
    times = []
    while True:
        if i % 100 == 0:
            print(f"Value Iteration #{i}")
        st_step = time.time()
        i += 1
        delta = 0  # checker for improvements across states
        for state in range(env.observation_space.n):
            act_values = one_step_lookahead(env, state, V)  # lookahead one step
            best_act_value = np.max(act_values)  # get best action value
            delta = max(delta, np.abs(best_act_value - V[state]))  # find max delta across all states
            V[state] = best_act_value  # update value to best action value
        times.append(time.time()-st_step)
        if delta < theta:  # if max improvement less than threshold
            break

    plt.figure()
    plt.plot(times)
    plt.show()

    policy = np.zeros([env.observation_space.n, env.action_space.n])
    for state in range(env.observation_space.n):  # for all states, create deterministic policy
        act_val = one_step_lookahead(env, state, V)
        best_action = np.argmax(act_val)
        policy[state][best_action] = 1

    # Implement!
    return Policy(policy=policy
                  , q_table=V
                  , name="Value Iteration"
                  , iterations_to_converge=i
                  , time_to_converge=time.time()-st)


def count_steps(env, policy, max_steps=1000):
    state = env.reset()
    counter = 0
    penalties = 0
    reward = None
    while reward != 20 and reward != 1 and counter < max_steps:
        state, reward, done, info = env.step(np.argmax(policy[state]))
        if done:
            env.reset()
            penalties +=1
        counter += 1

    return counter, penalties


def Q_learning_train(env, alpha, gamma, epsilon, episodes, dataset_name, epsilons=None):
    """Q Learning Algorithm with epsilon greedy

    Args:
        env: Environment
        alpha: Learning Rate --> Extent to which our Q-values are being updated in every iteration.
        gamma: Discount Rate --> How much importance we want to give to future rewards
        epsilon: Probability of selecting random action instead of the 'optimal' action
        episodes: No. of episodes to train on

    Returns:
        Q-learning Trained policy

    """

    """Training the agent"""

    # For plotting metrics
    metrics = {}
    epsilon_decay = 0.99
    epsilons = [x / 10 for x in range(1, 11, 2)] if not epsilons else epsilons
    policy = None
    q_table = None
    max_rewards = Metric()

    for epsilon in epsilons:
        print(f"Epsilon {epsilon}")
        metrics[epsilon] = {}
        all_epochs = []
        all_penalties = []
        all_rewards = []
        all_times = []
        experiment_epsilon = epsilon

        # Initialize Q table of 500 x 6 size (500 states and 6 actions) with all zeroes
        q_table = np.zeros([env.observation_space.n, env.action_space.n])

        for i in range(1, episodes + 1):
            st = time.time()
            experiment_epsilon *= epsilon_decay
            if i % 100 == 0:
                print(f"Episode {i}")

            state = env.reset()

            epochs, penalties, reward, total_reward = 0, 0, 0, 0

            while reward != 20 and reward != 5 and epochs <= 10000:
                if epochs % 1000 == 0:
                    print(f"Epoch #{epochs}")

                if random.uniform(0, 1) < experiment_epsilon:
                    action = env.action_space.sample()  # Explore action space randomly
                else:
                    action = np.argmax(q_table[state])  # Exploit learned values by choosing optimal values

                next_state, reward, done, info = env.step(action)
                if done:
                    env.reset()
                    if reward == 1:
                        reward = 5

                    if reward == 0:
                        penalties += 1
                        reward = -5
                else:
                    if reward == 0:
                        reward = -1

                old_value = q_table[state, action]
                next_max = np.max(q_table[next_state])

                new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                q_table[state, action] = new_value

                if reward == -10:
                    penalties += 1

                state = next_state
                epochs += 1
                total_reward += reward

            max_rewards.track(reward=total_reward, episode=i, epsilon=epsilon, step=epochs)
            all_rewards.append(total_reward)
            all_epochs.append(epochs)
            all_penalties.append(penalties)
            all_times.append(time.time() - st)

        metrics[epsilon]['epochs'] = all_epochs
        metrics[epsilon]['penalties'] = all_penalties
        metrics[epsilon]['rewards'] = all_rewards
        metrics[epsilon]['times'] = all_times

        # Start with a random policy
        policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n

        for state in range(env.observation_space.n):  # for each states
            best_act = np.argmax(q_table[state])  # find best action
            policy[state] = np.eye(env.action_space.n)[best_act]  # update

    print(max_rewards)

    for metric in ['rewards', 'epochs', 'penalties', 'times']:
        plt.figure()
        title = metric + ' analysis'
        plt.title(title)
        plt.xlabel('Episodes')
        plt.ylabel(metric)
        for k, v in metrics.items():
            plt.plot(smooth(v[metric], window=500), label=f"epsilon={k}")
        plt.legend()
        filename = '%s_%s_%s' % (metric, 'qlearning', dataset_name)
        chart_path = 'report/images/%s.png' % filename
        print(chart_path)
        plt.savefig(chart_path)
        plt.close()
        processor = Processor()
        processor.latex_subgraph(dataset=env, fig=filename, caption=metric, filename=filename)

    return Policy(policy=policy, q_table=q_table, name="Q Learning", iterations_to_converge=0)


# def view_policy(env, policy):
def view_policy(title, policy, map_desc, color_map, direction_map):
    policy = np.argmax(policy, axis=-1).reshape(4, 4)

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
    chartpath = 'report/images/' + clean_title + str('.png')
    print(chartpath)
    plt.savefig(chartpath)
    plt.close()
    processor = Processor()
    processor.latex_subgraph(dataset='frozenlake', fig=clean_title, caption=f'frozenlake {clean_title}',
                             filename=clean_title)

    return (plt)


"""A set of common utilities used within the environments. These are
not intended as API functions, and will not remain stable over time.
"""

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)


def colorize(string, color, bold=False, highlight=False):
    """Return string surrounded by appropriate terminal color codes to
    print colorized text.  Valid colors: gray, red, green, yellow,
    blue, magenta, cyan, white, crimson
    """

    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    attrs = ';'.join(attr)
    return '\x1b[%sm%s\x1b[0m' % (attrs, string)


def render(self):
    outfile = sys.stdout

    out = self.desc.copy().tolist()
    out = [[c.decode('utf-8') for c in line] for line in out]
    taxi_row, taxi_col, pass_idx, dest_idx = self.decode(self.s)

    def ul(x):
        return "_" if x == " " else x

    if pass_idx < 4:
        out[1 + taxi_row][2 * taxi_col + 1] = colorize(
            out[1 + taxi_row][2 * taxi_col + 1], 'yellow', highlight=True)
        pi, pj = self.locs[pass_idx]
        out[1 + pi][2 * pj + 1] = colorize(out[1 + pi][2 * pj + 1], 'blue', bold=True)
    else:  # passenger in taxi
        out[1 + taxi_row][2 * taxi_col + 1] = colorize(
            ul(out[1 + taxi_row][2 * taxi_col + 1]), 'green', highlight=True)

    di, dj = self.locs[dest_idx]
    out[1 + di][2 * dj + 1] = colorize(out[1 + di][2 * dj + 1], 'magenta')
    outfile.write("\n".join(["".join(row) for row in out]) + "\n")
    if self.lastaction is not None:
        outfile.write("  ({})\n".format(["South", "North", "East", "West", "Pickup", "Dropoff"][self.lastaction]))
    else:
        outfile.write("\n")


def compare_policies(first, second):
    print(f"Comparing {first.name} and {second.name}")
    return same_policies(first.policy, second.policy)


def same_policies(first, second):
    if first is None:
        return False

    if second is None:
        return False

    for x in range(len(first[0])):
        if not (first[0][x] == second[0][x]).all():
            print("Not the same Policy")
            return False
    print("Same Policy")
    return True


def run(dataset_name, episodes=100000, env=None, epsilons=None):
    plt.figure()
    processor = Processor()

    env = gym.make(dataset_name) if not env else env
    print(env.spec.id)
    title = run_random_agent(dataset_name, env, processor)

    env.reset()
    pol_iter_policy = run_pi_agent(dataset_name, env, processor, title)

    env.reset()
    val_iter_policy = run_vi_agent(dataset_name, env, processor, title)

    env.reset()
    Q_learn_pol = run_qlearning_agent(dataset_name, env, episodes, processor, title, epsilons=epsilons)

    compare_policies(pol_iter_policy, Q_learn_pol)
    compare_policies(Q_learn_pol, val_iter_policy)
    compare_policies(pol_iter_policy, val_iter_policy)
    if 'Frozen' in env.spec.id:
        view_policy(title=f"Policy Iterations - {dataset_name}", policy=pol_iter_policy.policy,
                    map_desc=env.unwrapped.desc,
                    color_map=colors_lake(), direction_map=directions_lake())
        view_policy(title=f"Value Iterations - {dataset_name}", policy=val_iter_policy.policy,
                    map_desc=env.unwrapped.desc,
                    color_map=colors_lake(), direction_map=directions_lake())
        view_policy(title=f"Q Learning - {dataset_name}", policy=Q_learn_pol.policy, map_desc=env.unwrapped.desc,
                    color_map=colors_lake(), direction_map=directions_lake())


def run_qlearning_agent(dataset_name, env, episodes, processor, title, epsilons=None):
    plt.figure()
    Q_learn_pol = Q_learning_train(env=env, alpha=0.2, gamma=0.95, epsilon=0.95, episodes=episodes, epsilons=epsilons, dataset_name=dataset_name)
    Q_counts = []
    penalties = []
    for i in range(100):
        if i % 10 == 0:
            print(i)
        count, penalty = count_steps(env=env, policy=Q_learn_pol.policy)
        Q_counts.append(count)
        penalties.append(penalty)
    print("An agent using a policy which has been improved using Q-learning takes about an average of " + str(
        int(np.mean(Q_counts)))
          + " steps to successfully complete its mission.")

    sns.distplot(Q_counts)
    fig = re.sub('[^0-9a-zA-Z]+', '_', title.lower())
    plt.title("%s" % title)
    filename = '%s_%s_%s' % ('steps', 'qlearning', dataset_name)
    chart_path = 'report/images/%s.png' % filename
    plt.savefig(chart_path)
    plt.close()
    processor.latex_subgraph(dataset=dataset_name, fig=filename, caption="Q Learning", filename=filename)
    processor.latex_end_figure(caption=title, fig=fig)

    plt.figure()
    sns.distplot(penalties)
    title = "Distribution of penalties"
    fig = re.sub('[^0-9a-zA-Z]+', '_', title.lower())
    plt.title("%s" % title)
    filename = '%s_%s_%s' % ('penalties_steps', 'qlearning', dataset_name)
    chart_path = 'report/images/%s.png' % filename
    plt.savefig(chart_path)
    print(chart_path)
    plt.close()
    processor.latext_start_figure()
    processor.latex_subgraph(dataset=dataset_name, fig=filename, caption='Q Learning', filename=filename)
    return Q_learn_pol


def run_vi_agent(dataset_name, env, processor, title):
    plt.figure()
    val_iter_policy = value_iteration(env=env, discount_factor=0.99, theta=0.0001)
    val_counts = []
    penalties = []
    for i in range(100):
        if i % 10 == 0:
            print(i)
        count, penalty = count_steps(env=env, policy=val_iter_policy.policy)
        val_counts.append(count)
        penalties.append(penalty)
    print(f"Value Iterations took {val_iter_policy.iterations_to_converge} steps to converge.")
    print(f"Value Iterations took {val_iter_policy.time_to_converge} seconds to converge.")
    print("An agent using a policy which has been value-iterated takes about an average of " + str(
        int(np.mean(val_counts)))
          + " steps to successfully complete its mission.")
    sns.distplot(val_counts)
    fig = re.sub('[^0-9a-zA-Z]+', '_', title.lower())
    plt.title("%s" % title)
    filename = '%s_%s_%s' % ('steps', 'value', dataset_name)
    chart_path = 'report/images/%s.png' % filename
    plt.savefig(chart_path)
    plt.close()
    processor.latex_subgraph(dataset=dataset_name, fig=filename, caption="VI", filename=filename)

    plt.figure()
    sns.distplot(penalties)
    title = "Distribution of penalties"
    fig = re.sub('[^0-9a-zA-Z]+', '_', title.lower())
    plt.title("%s" % title)
    filename = '%s_%s_%s' % ('penalties_steps', 'vi', dataset_name)
    chart_path = 'report/images/%s.png' % filename
    plt.savefig(chart_path)
    print(chart_path)
    plt.close()
    processor.latext_start_figure()
    processor.latex_subgraph(dataset=dataset_name, fig=filename, caption='VI', filename=filename)
    return val_iter_policy


def run_pi_agent(dataset_name, env, processor, title):
    plt.figure()
    pol_iter_policy = policy_iteration(env, policy_eval, discount_factor=0.99)
    pol_counts = []
    penalties = []
    for i in range(100):
        if i % 10 == 0:
            print(i)
        count, penalty = count_steps(env=env, policy=pol_iter_policy.policy)
        pol_counts.append(count)
        penalties.append(penalty)

    print(f"Policy Iterations took {pol_iter_policy.iterations_to_converge} steps to converge.")
    print(f"Policy Iterations took {pol_iter_policy.time_to_converge} seconds to converge.")
    print("An agent using a policy which has been improved using policy-iterated takes about an average of " + str(
        int(np.mean(pol_counts)))
          + " steps to successfully complete its mission.")
    sns.distplot(pol_counts)
    fig = re.sub('[^0-9a-zA-Z]+', '_', title.lower())
    plt.title("%s" % title)
    filename = '%s_%s_%s' % ('steps', 'policy', dataset_name)
    chart_path = 'report/images/%s.png' % filename
    plt.savefig(chart_path)
    plt.close()
    processor.latex_subgraph(dataset=dataset_name, fig=filename, caption="PI", filename=filename)

    plt.figure()
    sns.distplot(penalties)
    title = "Distribution of penalties"
    fig = re.sub('[^0-9a-zA-Z]+', '_', title.lower())
    plt.title("%s" % title)
    filename = '%s_%s_%s' % ('penalties_steps', 'policy', dataset_name)
    chart_path = 'report/images/%s.png' % filename
    plt.savefig(chart_path)
    print(chart_path)
    plt.close()
    processor.latext_start_figure()
    processor.latex_subgraph(dataset=dataset_name, fig=filename, caption='PI', filename=filename)
    return pol_iter_policy


def run_random_agent(dataset_name, env, processor):
    counts = []
    penalties = []
    for i in range(100):
        if i % 10 == 0:
            print(i)
        count, penalty = random_policy_steps_count(env=env)
        counts.append(count)
        penalties.append(penalty)

    print("An agent using Random search takes about an average of " + str(int(np.mean(counts)))
          + " steps to successfully complete its mission.")
    print("An agent using Random search is penalized " + str(int(np.mean(penalties))) + " times.")
    plt.figure()
    sns.distplot(counts)
    title = "Distribution of number of steps needed"
    fig = re.sub('[^0-9a-zA-Z]+', '_', title.lower())
    plt.title("%s" % title)
    filename = '%s_%s_%s' % ('steps', 'random', dataset_name)
    chart_path = 'report/images/%s.png' % filename
    plt.savefig(chart_path)
    print(chart_path)
    plt.close()
    processor.latext_start_figure()
    processor.latex_subgraph(dataset=dataset_name, fig=filename, caption='Random', filename=filename)

    plt.figure()
    sns.distplot(penalties)
    title = "Distribution of penalties"
    fig = re.sub('[^0-9a-zA-Z]+', '_', title.lower())
    plt.title("%s" % title)
    filename = '%s_%s_%s' % ('penalties_steps', 'random', dataset_name)
    chart_path = 'report/images/%s.png' % filename
    plt.savefig(chart_path)
    print(chart_path)
    plt.close()
    processor.latext_start_figure()
    processor.latex_subgraph(dataset=dataset_name, fig=filename, caption='Random', filename=filename)
    return title


def run_experiments_part2():
    # run(dataset_name='Taxi-v1', episodes=5000)
    run(dataset_name='FrozenLake-v0', episodes=5000)

    # random_map = generate_random_map(size=40, p=0.8)
    # env = gym.make("FrozenLake-v0", desc=random_map)
    # run(dataset_name='FrozenLake-40x40', episodes=5000, env=env, epsilons=[0.99])


if __name__ == '__main__':
    run_experiments_part2()
