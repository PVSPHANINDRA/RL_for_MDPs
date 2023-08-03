import sys

from learningAlgorithms import *
from acrobot_online_sarsa import *
from util import *
from gridWorldMDP import *
from mountainCar import *
from cartPole import *
from acrobot import *

if __name__ == '__main__':
    if sys.argv[1] == '1':  # Grid World
        steps_taken = []
        MSEs = []
        q_values = []
        itr_count = 1000
        for i in range(5):
            print(i)
            _d = 9
            _alpha = 0.1
            _e = 0.1
            _lambda = 0.0
            feature_function = initialize_feature_function_for_grid_world(_d)
            W = initialize_w(_d)
            cur_steps_taken, cur_MSEs, q = true_online_SARSA_for_grid_world(itr_count, feature_function, W, _alpha, _e, _lambda, _d, gamma_for_grid_world)
            steps_taken.append(cur_steps_taken)
            MSEs.append(cur_MSEs)
            q_values.append(q)
        x = [i for i in range(1, itr_count + 1, 1)]
        plot_the_episode_vs_timestamps_graph(x, steps_taken, 'SARSA', 'Grid World')
        plot_the_mse_episodes_graph(x, MSEs, 'SARSA')
        avg_q = compute_avg_action_value_function_for_grid_world(q_values)
        estimated_optimal_policy = estimate_policy_from_action_value_function_for_grid_world(avg_q)
        print('Greedy Policy with respect to the q-values learned by SARSA')
        display_the_policy(estimated_optimal_policy)
    elif sys.argv[1] == '2':  # Mountain Car
        steps_taken = []
        observed_rewards = []
        itr_count = 1500
        graph_step_size = 1
        _d = 7
        _alpha = 0.00001
        _e = 0.001
        _lambda = 0.95
        mountain_car_gamma = 1.0

        v_features = getFourierSeries(_d, state_variables_for_mountain_car)
        _d = len(v_features)
        for i in range(5):
            print(i)
            W = initialize_w_for_mountain_car(_d)
            cur_steps_taken, cur_observed_rewards = true_online_SARSA_for_mountain_car(itr_count, graph_step_size, W, _alpha, _e, _lambda, _d,
                                                                                       mountain_car_gamma, v_features)
            steps_taken.append(cur_steps_taken)
            observed_rewards.append(cur_observed_rewards)
        x = [i for i in range(graph_step_size, itr_count + graph_step_size, graph_step_size)]
        plot_the_episode_vs_timestamps_graph(x, steps_taken, 'True Online SARSA(\u03BB)', 'Mountain Car', 'Time steps', 'Episodes')
        plot_the_episode_vs_steps_need_by_mountain_car_graph(x, observed_rewards, 'True Online SARSA(\u03BB)', 'Mountain Car', 'Episodes',
                                                             '#Steps to reach goal')
    elif sys.argv[1] == '3':  # Cart Pole
        steps_taken = []
        observed_rewards = []
        # Hyper-params
        itr_count = 2000
        graph_step_size = 1
        _d = 5
        _alpha = 0.00001
        _e = 0.0001
        _lambda = 0.9
        cart_pole_gamma = 1.0

        v_features = getFourierSeries(_d, state_variables_for_cart_pole)
        _d = len(v_features)
        print('itr_count:', itr_count, '_alpha', _alpha, '_e', _e, '_lambda', _lambda, '_d', _d)
        for i in range(3):
            print(i)
            W = initialize_w_as_array(_d, len(actions_available_for_cart_pole), True)
            cur_steps_taken, cur_observed_rewards = true_online_SARSA_for_cart_pole(itr_count, graph_step_size, W, _alpha, _e, _lambda, _d,
                                                                                    cart_pole_gamma, v_features, actions_available_for_cart_pole)
            steps_taken.append(cur_steps_taken)
            observed_rewards.append(cur_observed_rewards)
        x = [i for i in range(graph_step_size, itr_count + graph_step_size, graph_step_size)]
        plot_the_episode_vs_timestamps_graph(x, steps_taken, 'True Online SARSA(\u03BB)', 'Cart Pole', 'Time steps', 'Episodes')
        plot_the_episode_vs_steps_need_by_mountain_car_graph(x, observed_rewards, 'True Online SARSA(\u03BB)', 'Cart Pole', 'Episodes',
                                                             '#Number of Actions')
    elif sys.argv[1] == '4':  # Acrobot
        steps_taken = []
        observed_rewards = []
        # Hyperparams
        itr_count = 1500
        graph_step_size = 1
        _d = 1
        _alpha = 0.00001
        _e = 0.001
        _lambda = 0.95
        acrobot_gamma = 1.0

        v_features = getFourierSeries(_d, state_variables_for_acrobot)
        _d = len(v_features)
        for i in range(20):
            print(i)
        W = initialize_w_as_array(_d, len(actions_available_for_acrobot))
        cur_steps_taken, cur_observed_rewards = true_online_SARSA_for_acrobot(itr_count, graph_step_size, W, _alpha, _e, _lambda, _d,
                                                                              acrobot_gamma, v_features, actions_available_for_acrobot)
        steps_taken.append(cur_steps_taken)
        observed_rewards.append(cur_observed_rewards)
        x = [i for i in range(graph_step_size, itr_count + graph_step_size, graph_step_size)]
        plot_the_episode_vs_timestamps_graph(x, steps_taken, 'True Online SARSA(\u03BB)', 'Acrobot', 'Time steps', 'Episodes')
        plot_the_episode_vs_steps_need_by_mountain_car_graph(x, observed_rewards, 'True Online SARSA(\u03BB)', 'Acrobot', 'Episodes',
                                                             '#Steps to reach goal')
