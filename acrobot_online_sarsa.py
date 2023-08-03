import math

import numpy as np
import gym

from util import *
from acrobot import *


def true_online_SARSA_for_acrobot(itr_count, graph_step_size, W, _alpha, _e, _lambda, _d, _gamma, v_features, actions):
    env = gym.make('Acrobot-v1')

    steps_taken = []
    timestamp = 0
    rewards_observed = []

    for itr in range(1, itr_count + 1, 1):
        observation = env.reset()[0]

        X = compute_state_feature_for_acrobot(observation[0], observation[1], observation[2], observation[3], observation[4], observation[5], v_features)
        a = choose_action_using_e_greedy(
            {0: np.dot(np.transpose(W[:, [0]]), X)[0][0],
             1: np.dot(np.transpose(W[:, [1]]), X)[0][0],
             2: np.dot(np.transpose(W[:, [2]]), X)[0][0]}, _e)
        Z = np.zeros((_d, len(actions)))
        Q_old = 0

        timestamp += 1
        r = 0
        while True:
            observation_1, reward, terminated, truncated, info = env.step(a)
            r += 1

            X_1 = compute_state_feature_for_acrobot(observation_1[0], observation_1[1], observation_1[2], observation_1[3], observation[4], observation[5],
                                                    v_features)
            a1 = choose_action_using_e_greedy(
                {0: np.dot(np.transpose(W[:, [0]]), X_1)[0][0],
                 1: np.dot(np.transpose(W[:, [1]]), X_1)[0][0],
                 2: np.dot(np.transpose(W[:, [2]]), X_1)[0][0]}, _e)

            Q = compute_Q(W, X, actions.index(a))
            Q_1 = compute_Q(W, X_1, actions.index(a1))

            delta = reward + _gamma * Q_1 - Q

            index = actions.index(a)
            Z[:, [index]] = _gamma * _lambda * Z[:, [index]] + (1 - _alpha * _gamma * _lambda * np.dot(np.transpose(Z[:, [index]]), X)) * X
            W[:, [index]] = W[:, [index]] + _alpha * (delta + Q - Q_old) * Z[:, [index]] - _alpha * (Q - Q_old) * X

            Q_old = Q_1
            X = X_1
            a = a1
            observation = observation_1

            timestamp += 1
            if terminated:
                break
            if r > 500:
                break
        # print(itr, r)
        # if itr % 100 == 0:
        #     _e /= 10
        if itr % graph_step_size == 0:
            steps_taken.append(timestamp - 1)
            rewards_observed.append(r)
    return steps_taken, rewards_observed


def compute_state_feature_for_acrobot(c_t1, s_t1, c_t2, s_t2, a_t1, a_t2, v_features):
    c_t1 = (c_t1 + 1) / 2
    s_t1 = (s_t1 + 1) / 2
    c_t2 = (c_t2 + 1) / 2
    s_t2 = (s_t2 + 1) / 2
    a_t1 = (a_t1 + 12.57) / 25.14
    a_t2 = (a_t2 + 28.27) / 56.54

    return np.cos(np.dot(v_features, np.resize(np.asarray([c_t1, s_t1, c_t2, s_t2, a_t1, a_t2]), (6, 1))) * math.pi)
