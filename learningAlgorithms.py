import math

import numpy as np
import gym

from util import *
from gridWorldMDP import *
from mountainCar import *
from cartPole import *


def true_online_SARSA_for_cart_pole(itr_count, graph_step_size, W, _alpha, _e, _lambda, _d, _gamma, v_features, actions):
    env = gym.make('CartPole-v1')

    steps_taken = []
    timestamp = 0
    rewards_observed = []

    step_size = 100
    _e_decay_rate = (_e - 0) / (itr_count / step_size)

    for itr in range(1, itr_count + 1, 1):
        observation = env.reset()[0]

        X = compute_state_feature_for_cart_pole(observation[0], observation[1], observation[2], observation[3], v_features)
        a = choose_action_using_e_greedy(
            {0: np.dot(np.transpose(W[:, [0]]), X)[0][0],
             1: np.dot(np.transpose(W[:, [1]]), X)[0][0]}, _e)
        Z = np.zeros((_d, len(actions)))
        Q_old = 0

        timestamp += 1
        r = 0
        while True:
            observation_1, reward, terminated, truncated, info = env.step(a)
            # max_v, max_theta_dot = max(max_v, observation_1[1]), max(max_theta_dot, observation_1[3])
            if observation_1[1] > MAX_V_FOR_CART_POLE or observation_1[1] < -1 * MAX_V_FOR_CART_POLE:
                sys.exit('observed greater v value {}'.format(observation_1[1]))
            if observation_1[3] > MAX_THETA_DOT_FOR_CART_POLE or observation_1[3] < -1 * MAX_THETA_DOT_FOR_CART_POLE:
                sys.exit('observed greater theta_dot value {}'.format(observation_1[3]))
            r += 1

            X_1 = compute_state_feature_for_cart_pole(observation_1[0], observation_1[1], observation_1[2], observation_1[3], v_features)
            a1 = choose_action_using_e_greedy(
                {0: np.dot(np.transpose(W[:, [0]]), X_1)[0][0],
                 1: np.dot(np.transpose(W[:, [1]]), X_1)[0][0]}, _e)

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
        if itr % step_size == 0:
            _e /= 10
        if itr % graph_step_size == 0:
            steps_taken.append(timestamp - 1)
            rewards_observed.append(r)
    # print(max_v, max_theta_dot)
    return steps_taken, rewards_observed


def true_online_SARSA_for_mountain_car(itr_count, graph_step_size, W, _alpha, _e, _lambda, _d, _gamma, v_features):
    steps_taken = []
    timestamp = 0
    rewards_observed = []

    for itr in range(1, itr_count + 1, 1):
        x_t, v_t = random.uniform(mountain_car_x_range[0], mountain_car_x_range[1]), random.uniform(mountain_car_v_range[0],
                                                                                                    mountain_car_v_range[1])
        while x_t == mountain_car_terminal_x:
            x_t = random.uniform(mountain_car_x_range[0], mountain_car_x_range[1])
        x_t, v_t = for_mountain_car_clip_x_v(x_t, v_t)

        X = compute_state_feature_for_mountain_car(x_t, v_t, v_features)
        a = choose_action_using_e_greedy(
            {-1: np.dot(np.transpose(W[0]), X)[0][0],
             0: np.dot(np.transpose(W[1]), X)[0][0],
             1: np.dot(np.transpose(W[2]), X)[0][0]}, _e)
        Z = [np.zeros((_d, 1)), np.zeros((_d, 1)), np.zeros((_d, 1))]
        Q_old = 0

        timestamp += 1
        r = 0
        while x_t != mountain_car_terminal_x:
            r -= 1
            x_t1, v_t1 = calculate_the_next_x_v(x_t, v_t, a)

            X_1 = compute_state_feature_for_mountain_car(x_t1, v_t1, v_features)
            a1 = choose_action_using_e_greedy(
                {-1: np.dot(np.transpose(W[0]), X_1)[0][0],
                 0: np.dot(np.transpose(W[1]), X_1)[0][0],
                 1: np.dot(np.transpose(W[2]), X_1)[0][0]}, _e)

            Q = compute_Q_for_mountain_car(W, X, a)
            Q_1 = compute_Q_for_mountain_car(W, X_1, a1)

            delta = r + _gamma * Q_1 - Q
            if a == -1:
                Z[0] = _gamma * _lambda * Z[0] + (1 - _alpha * _gamma * _lambda * np.dot(np.transpose(Z[0]), X)) * X
                W[0] = W[0] + _alpha * (delta + Q - Q_old) * Z[0] - _alpha * (Q - Q_old) * X
            elif a == 0:
                Z[1] = _gamma * _lambda * Z[1] + (1 - _alpha * _gamma * _lambda * np.dot(np.transpose(Z[1]), X)) * X
                W[1] = W[1] + _alpha * (delta + Q - Q_old) * Z[1] - _alpha * (Q - Q_old) * X
            elif a == 1:
                Z[2] = _gamma * _lambda * Z[2] + (1 - _alpha * _gamma * _lambda * np.dot(np.transpose(Z[2]), X)) * X
                W[2] = W[2] + _alpha * (delta + Q - Q_old) * Z[2] - _alpha * (Q - Q_old) * X
            else:
                sys.exit('Should Never Reach Here 1')

            Q_old = Q_1
            X = X_1
            a = a1

            x_t, v_t = x_t1, v_t1
            timestamp += 1
            if r == -1000:
                break
        # print(itr, r)
        if itr % graph_step_size == 0:
            steps_taken.append(timestamp - 1)
            rewards_observed.append(int(abs(math.ceil(r))))
    return steps_taken, rewards_observed


def true_online_SARSA_for_grid_world(itr_count, feature_function, W, _alpha, _e, _lambda, _d, _gamma):
    steps_taken = []
    timestamp = 0
    MSEs = []

    for itr in range(1, itr_count + 1, 1):
        timestamp += 1

        s = selectRandomInitialState_for_grid_world()
        a = choose_action_using_e_greedy(compute_q_of_state_from_feature_function(feature_function, W, s), _e)
        X = feature_function[s][a]
        Z = np.zeros((_d, 1))
        Q_old = 0
        while s not in terminal_states_for_grid_world:
            s1 = calculate_next_state_for_grid_world(s, a)
            r = rewards_for_grid_world[s1]
            a1 = choose_action_using_e_greedy(compute_q_of_state_from_feature_function(feature_function, W, s), _e)

            X_1 = feature_function[s1][a1]
            Q = np.dot(np.transpose(W), X)[0][0]
            Q_1 = np.dot(np.transpose(W), X_1)[0][0]

            delta = r + _gamma * Q_1 - Q
            Z = _gamma * _lambda * Z + (1 - _alpha * _gamma * _lambda * np.dot(np.transpose(Z), X)) * X
            W = W + _alpha * (delta + Q - Q_old) * Z - _alpha * (Q - Q_old) * X

            Q_old = Q
            X = X_1

            s, a = s1, a1
            timestamp += 1
        q = compute_q_from_feature_function(feature_function, W)
        steps_taken.append(timestamp - 1)
        MSEs.append(computeMSE(populate_the_value_function_for_grid_world(q, _e), optimal_policy_v_for_grid_world))
    return steps_taken, MSEs, compute_q_from_feature_function(feature_function, W)
