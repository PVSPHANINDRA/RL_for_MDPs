from collections import defaultdict
import random
import sys
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import math

import sympy
from sympy import symbols

from gridWorldMDP import *
from mountainCar import *
from cartPole import *


def initialize_q_for_grid_world():
    q = defaultdict(lambda: {'AU': 1e2, 'AD': 1e2, 'AR': 1e2, 'AL': 1e2})
    for state in terminal_states_for_grid_world:
        for action in q[state]:
            q[state][action] = 0.0
    return q


def compute_q_from_feature_function(feature_function, W):
    q = defaultdict(lambda: {'AU': 0.0, 'AD': 0.0, 'AR': 0.0, 'AL': 0.0})
    for state in possible_initial_states_for_grid_world:
        for action in q[state]:
            q[state][action] = np.dot(np.transpose(W), feature_function[state][action])[0][0]
    return q


def initialize_e_traces_for_grid_world():
    return defaultdict(lambda: {'AU': 0.0, 'AD': 0.0, 'AR': 0.0, 'AL': 0.0})


def initialize_feature_function_for_grid_world(d, order=3):
    feature_function = defaultdict(
        lambda: {'AU': np.random.rand(d, 1), 'AD': np.random.rand(d, 1), 'AR': np.random.rand(d, 1), 'AL': np.random.rand(d, 1)})
    for state in possible_initial_states_for_grid_world:
        for action in feature_function[state]:
            x, y = (state[0] - 0) / 4, (state[1] - 0) / 4
            feature_function[state][action] = calculate(x, y)
    for state in terminal_states_for_grid_world:
        for action in feature_function[state]:
            feature_function[state][action] = np.zeros((9, 1))
    return feature_function


def calculate(row, col):
    return np.resize(np.asarray([1, math.cos(math.pi * row), math.cos(math.pi * 2 * row), math.cos(math.pi * col), math.cos(math.pi * 2 * col),
                                 math.cos(math.pi * (row + col)), math.cos(math.pi * (row + 2 * col)), math.cos(math.pi * (2 * row + col)),
                                 math.cos(math.pi * (2 * row + 2 * col))]), (9, 1))


def initialize_w(d):
    return np.random.rand(d, 1)


def selectRandomInitialState_for_grid_world():
    return random.choice(possible_initial_states_for_grid_world)


def compute_q_of_state_from_feature_function(feature_function, W, s):
    q = {}
    for action in feature_function[s]:
        q[action] = np.dot(np.transpose(W), feature_function[s][action])[0][0]
    return q


def choose_action_using_e_greedy(q, e):
    probabilities = estimate_policy_for_given_state(q, e)
    pr = random.random()
    t = 0
    for a in probabilities:
        if t < pr <= t + probabilities[a]:
            return a
        t += probabilities[a]
    sys.exit('Should Never Come Here')  # Should Never Come Here


def estimate_policy_for_given_state(q, e):
    d = defaultdict(lambda: set())
    for action in q:
        d[q[action]].add(action)
    cur_max = float('-inf')
    for k in d:
        if k > cur_max:
            cur_max = k
    A_star = d[cur_max]
    probabilities = {}
    for a in q:
        if a in A_star:
            probabilities[a] = ((1 - e) / (len(A_star))) + (e / len(q))
        else:
            probabilities[a] = e / len(q)
    return probabilities


def calculate_next_state_for_grid_world(s, action):
    x, y = s[0], s[1]
    x1, y1 = x, y
    executed_action = action_performed_for_grid_world(action)
    if executed_action is not None:
        if executed_action == 'AU':
            x1 = max(0, x - 1)
            if (x1 == 2 and y == 2) or (x1 == 3 and y == 2):
                x1 = x
        elif executed_action == 'AD':
            x1 = min(m_for_grid_world - 1, x + 1)
            if (x1 == 2 and y == 2) or (x1 == 3 and y == 2):
                x1 = x
        elif executed_action == 'AR':
            y1 = min(n_for_grid_world - 1, y + 1)
            if (x == 2 and y1 == 2) or (x == 3 and y1 == 2):
                y1 = y
        elif executed_action == 'AL':
            y1 = max(0, y - 1)
            if (x == 2 and y1 == 2) or (x == 3 and y1 == 2):
                y1 = y
        else:
            sys.exit('Invalid action encountered in calculate_next_state()')
    s1 = (x1, y1)
    return s1


def action_performed_for_grid_world(action):
    pr = random.random()
    if action == 'AU':
        if 0 < pr <= 0.8:
            return 'AU'
        elif 0.8 < pr <= 0.85:
            return 'AR'
        elif 0.85 < pr <= 0.9:
            return 'AL'
        else:
            return None
    elif action == 'AL':
        if 0 < pr <= 0.8:
            return 'AL'
        elif 0.8 < pr <= 0.85:
            return 'AU'
        elif 0.85 < pr <= 0.9:
            return 'AD'
        else:
            return None
    elif action == 'AR':
        if 0 < pr <= 0.8:
            return 'AR'
        elif 0.8 < pr <= 0.85:
            return 'AD'
        elif 0.85 < pr <= 0.9:
            return 'AU'
        else:
            return None
    elif action == 'AD':
        if 0 < pr <= 0.8:
            return 'AD'
        elif 0.8 < pr <= 0.85:
            return 'AL'
        elif 0.85 < pr <= 0.9:
            return 'AR'
        else:
            return None
    else:
        sys.exit('Invalid action encountered in action_performed()')


def computeMSE(v1, v2):
    errors = []
    for i in range(m_for_grid_world):
        for j in range(n_for_grid_world):
            errors.append((v1[i][j] - v2[i][j]) ** 2)
    return np.mean(errors)


def populate_the_value_function_for_grid_world(q, e):
    v = [[0.0 for _ in range(n_for_grid_world)] for _ in range(m_for_grid_world)]
    for state in q:
        val = 0
        probabilities = estimate_policy_for_given_state(q[state], e)
        for action in probabilities:
            val += q[state][action] * probabilities[action]
        v[state[0]][state[1]] = val
    return v


def plot_the_episode_vs_timestamps_graph(episodes, steps_taken, algo, prob, x_label, y_label):
    x = np.mean(steps_taken, axis=0)
    y = episodes
    plt.plot(x, y)
    plt.title('{} ({})\nEpisodes Vs Time steps'.format(prob, algo))
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.show()


def plot_the_mse_episodes_graph(episodes, MSEs, algo):
    x = episodes
    y = np.mean(MSEs, axis=0)
    remove_indexes = int(math.floor(len(x) * 0.05))
    x = x[remove_indexes:]
    y = y[remove_indexes:]
    plt.plot(x, y)
    plt.title('(Gridworld {})\nMSE Vs Episodes'.format(algo))
    plt.ylabel('MSE')
    plt.xlabel('Episodes')
    plt.show()


def compute_avg_action_value_function_for_grid_world(q_values):
    avg_q = defaultdict(lambda: {'AU': 0.0, 'AD': 0.0, 'AR': 0.0, 'AL': 0.0})
    for q in q_values:
        for i in range(m_for_grid_world):
            for j in range(n_for_grid_world):
                for a in q[(i, j)]:
                    avg_q[(i, j)][a] += q[(i, j)][a]
    L = len(q_values)
    for s in avg_q:
        for a in avg_q[s]:
            avg_q[s][a] /= L
    return avg_q


def estimate_policy_from_action_value_function_for_grid_world(q):
    policy = [[None for _ in range(n_for_grid_world)] for _ in range(m_for_grid_world)]
    for i in range(m_for_grid_world):
        for j in range(n_for_grid_world):
            if (i, j) in terminal_states_for_grid_world:
                policy[i][j] = 'G'
            elif (i, j) in obstacle_states_for_grid_world:
                policy[i][j] = None
            else:
                cur_max = float('-inf')
                cur_action = None
                for action in q[(i, j)]:
                    if q[(i, j)][action] > cur_max:
                        cur_max = q[(i, j)][action]
                        cur_action = action
                policy[i][j] = cur_action
    return policy


def display_the_policy(policy):
    res = ''
    for i in range(len(policy)):
        res += ' '.join([get_symbol(j) for j in policy[i]]) + '\n'
    print(res)


def get_symbol(action):
    if action == 'AU':
        return '\u2191'
    elif action == 'AD':
        return '\u2193'
    elif action == 'AR':
        return '\u2192'
    elif action == 'AL':
        return '\u2190'
    elif action == 'G':
        return 'G'
    else:
        return ' '


# Mountain Car Methods

def for_mountain_car_clip_x_v(x, v):
    x = max(x, mountain_car_x_range[0])
    x = min(x, mountain_car_x_range[1])
    v = max(v, mountain_car_v_range[0])
    v = min(v, mountain_car_v_range[1])
    return x, v


def for_given_x_v_fetch_discretized_state_for_mountain_car(x, v):
    try:
        row = math.floor(round_off((x - mountain_car_x_range[0]) / mountain_car_x_step))
        col = math.floor(round_off((v - mountain_car_v_range[0]) / mountain_car_v_step))
        if row >= no_of_x_divisions:
            row -= 1
        if col >= no_of_v_divisions:
            col -= 1
        return mountain_car_states[row * no_of_v_divisions + col]
    except Exception as e:
        print(e, x, v)
        sys.exit('Should Never Reach Here func: for_given_x_v_fetch_discretized_state_for_mountain_car')


def calculate_the_next_x_v(x_t, v_t, a):
    v_t1 = v_t + 0.001 * a - 0.0025 * math.cos(3 * x_t)
    x_t1 = x_t + v_t1
    x_t1, v_t1 = for_mountain_car_clip_x_v(x_t1, v_t1)
    return x_t1, v_t1


def plot_the_episode_vs_steps_need_by_mountain_car_graph(episodes, steps, algo, prob, x_label, y_label):
    x = episodes
    y = np.mean(steps, axis=0)
    sd = np.std(steps, axis=0)
    plt.plot(x, y, 'k-')
    plt.fill_between(x, np.asarray(y) - np.asarray(sd), np.asarray(y) + np.asarray(sd))
    plt.title('{} ({})\nEpisodes Vs #Steps to reach goal'.format(prob, algo))
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.show()


def initialize_w_for_mountain_car(d):
    return [np.zeros((d, 1)), np.zeros((d, 1)), np.zeros((d, 1))]


def compute_state_feature_for_mountain_car(x, v, v_features):
    x = (x + 1.2) / 1.7
    v = (v + 0.07) / 0.14
    return np.cos(np.dot(v_features, np.resize(np.asarray([x, v]), (2, 1))) * math.pi)


def compute_Q_for_mountain_car(W, X, a):
    if a == -1:
        return np.dot(np.transpose(W[0]), X)[0][0]
    elif a == 0:
        return np.dot(np.transpose(W[1]), X)[0][0]
    elif a == 1:
        return np.dot(np.transpose(W[2]), X)[0][0]
    else:
        print('Should Never Reach Here')
        sys.exit('Should Never Reach Here')


def flatten(data):
    if isinstance(data, tuple):
        for x in data:
            yield from flatten(x)
    else:
        yield data


def getFourierSeries(order, state_variables):
    individual_arr = [[i for i in range(order + 1)] for var in state_variables]

    arr_prod = individual_arr[0]
    for i in range(1, len(individual_arr)):
        arr_prod = product(individual_arr[i], arr_prod)

    arr_prod = list(arr_prod)
    # print(arr_prod)

    flattened_arr = []
    for i in arr_prod:
        flattened_arr.append(list(flatten(i)))

    return np.array(flattened_arr).reshape(len(flattened_arr), len(state_variables))


# For Cart Pole


def initialize_w_as_array(d, action_count, random_flag=False):
    if random_flag:
        return np.random.rand(d, action_count)
    return np.zeros((d, action_count))


def compute_state_feature_for_cart_pole(x, v, theta, theta_dot, v_features):
    # Taking Max Value Observed in Experiments to normalize
    x = (x + 2.4) / 4.8
    v = (v + MAX_V_FOR_CART_POLE) / 2 * MAX_V_FOR_CART_POLE
    theta = (theta + 0.2095) / 0.418
    theta_dot = (theta_dot + MAX_THETA_DOT_FOR_CART_POLE) / 2 * MAX_THETA_DOT_FOR_CART_POLE
    return np.cos(np.dot(v_features, np.resize(np.asarray([x, v, theta, theta_dot]), (4, 1))) * math.pi)


def compute_Q(W, X, index):
    return np.dot(np.transpose(W[:, [index]]), X)[0][0]
