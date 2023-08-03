import numpy as np
from math import pi, exp, cos
import random
import matplotlib.pyplot as plt
from itertools import product
import gym

discount_parameter = 1
actions = [0, 1]
state_variables = ['x', 'v', 'angle', 'angular_velocity']
v_max_val = 4
ang_v_max_val = 2.5
state_variable_ranges = [[-4.8, 4.8],[-v_max_val, v_max_val],[-0.418, 0.418],[-ang_v_max_val, ang_v_max_val]]


def flatten(data):
    if isinstance(data, tuple):
        for x in data:
            yield from flatten(x)
    else:
        yield data


def getFourierSeries(order):
    if not len(state_variables):
        return []

    individual_arr = [[i for i in range(order+1)] for var in state_variables]
    product_arr = individual_arr[0]

    for i in range(1, len(individual_arr)):
        product_arr = product(individual_arr[i], product_arr)

    product_arr = list(product_arr)
    return np.array([list(flatten(i)) for i in product_arr])


def get_v_weights(order):
    total = int(pow(order+1, len(state_variables)))
    return np.array([0.0] * total).reshape(1, total)

def get_policy_weights(order):
    total = int(pow(order+1, len(state_variables)))
    return [np.array([0.0] * total).reshape(1, total) for i in actions]

def getPolicyEqOutput(policy_weights, policy_features, norm_state, cosineFlag=True):
    vals = []
    for i in range(len(actions)):
        weights = policy_weights[i]
        valSubstitutingInFeatures = np.matmul(policy_features, np.array(
            list(norm_state)).reshape(len(norm_state), 1))
        f_policy_features = np.cos(
            pi * valSubstitutingInFeatures) if cosineFlag else np.sin(pi * valSubstitutingInFeatures)
        vals.append(np.dot(weights, f_policy_features)[(0,0)])
    return vals

def getSoftMaxProbs(vals_arr, sigma):
    vals = np.array(vals_arr)
    return np.exp(sigma * vals - np.max(sigma * vals)) / np.exp(sigma * vals - np.max(sigma * vals)).sum()


def getEpsilonGreedyProbs(policyEqOutputArr, epsilon):
  max_action_value = max(policyEqOutputArr)
  max_action_indices = [i for i in range(len(policyEqOutputArr)) if policyEqOutputArr[i] == max_action_value]
  prob_arr=[]
  for _actionIndex in range(len(actions)):
    if _actionIndex in max_action_indices:
      prob = ((1- epsilon)/len(max_action_indices)) + (epsilon)/len(actions)
      prob_arr.append(prob)
    else:
      prob = (epsilon)/len(actions)
      prob_arr.append(prob)
  return prob_arr

def get_v_value(v_weights, v_features, norm_state, cosineFlag=True):
    valSubstitutingInFeatures = np.matmul(v_features, np.array(
        list(norm_state)).reshape(len(norm_state), 1))
    f_policy_features = np.cos(
        pi * valSubstitutingInFeatures) if cosineFlag else np.sin(pi * valSubstitutingInFeatures)
    val = np.dot(v_weights, f_policy_features)
    return val[(0,0)]

def get_v_feature_value(norm_state, v_features, cosineFlag= True):
    valSubstitutingInFeatures = np.matmul(v_features, np.array(
        list(norm_state)).reshape(len(norm_state), 1))
    f_policy_features = np.cos(
        pi * valSubstitutingInFeatures) if cosineFlag else np.sin(pi * valSubstitutingInFeatures)
    return np.transpose(f_policy_features)

def get_policy_feature_value(norm_state, policy_features, cosineFlag = True):
    valSubstitutingInFeatures = np.matmul(policy_features, np.array(
        list(norm_state)).reshape(len(norm_state), 1))
    f_policy_features = np.cos(
        pi * valSubstitutingInFeatures) if cosineFlag else np.sin(pi * valSubstitutingInFeatures)
    return np.transpose(f_policy_features)

def getNormValue(state, cosineFlag = True):
    res = []
    for i in range(len(state_variables)):
        state_range = state_variable_ranges[i]
        if cosineFlag:
            res.append((state[i] - state_range[0]) /
                       (state_range[1] - state_range[0]))
        else:
            res.append((2 * (state[i] - state_range[0]) /
                       (state_range[1]-state_range[0])) - 1)
    return tuple(res)

def runEpisode(policy_weights, v_weights, policy_features, v_features, policy_alpha, v_alpha, sigma, cosineFlag=True):
    env = gym.make('CartPole-v1')
    
    # initialising the state
    state = env.reset()[0]
    terminated = False
    timesteps = 0
    var_I = 1
    while True:
        if terminated or timesteps == 500: # terminal state
            return policy_weights, v_weights, timesteps

        timesteps += 1

         # norm the state
        norm_state = getNormValue(state, cosineFlag)

        # getting the policy equation output, using it to find the action by epsilon greedy or softmax
        policyEqOp = getPolicyEqOutput(
            policy_weights, policy_features, norm_state)
        actionProb = getSoftMaxProbs(policyEqOp, sigma)
        pickedAction = random.choices(actions, actionProb)[0] # picked action
        pickedActionIndex = actions.index(pickedAction)
        pickedActionProb = actionProb[pickedActionIndex]

        # getting next state by the picked action and its rewards
        next_state, reward, terminated, truncated, info = env.step(pickedAction)

        # normalizing the next state 
        norm_next_state = getNormValue(next_state, cosineFlag)

        # calculating the temporarl difference
        td_diff = reward + (discount_parameter * get_v_value(v_weights, v_features,
                            norm_next_state, cosineFlag)) - get_v_value(v_weights, v_features, norm_state, cosineFlag)
        v_weights += float(v_alpha * td_diff) * \
            get_v_feature_value(norm_state, v_features, cosineFlag)

        derviative = [1 - pickedActionProb if i == pickedActionIndex else -
                      1 * pickedActionProb for i in range(len(actions))]
        for i in range(len(actions)):
            policy_weights[i] += float(policy_alpha * td_diff * var_I) * \
                derviative[i] * \
                get_policy_feature_value(norm_state, policy_features, cosineFlag)

        var_I *= discount_parameter
        state = next_state

def plot_1(steps_taken, algo, prob, x_label, y_label, episodes_per_algo):
    y = [i for i in range(1,episodes_per_algo + 1)]
    plt.plot(steps_taken, y)
    plt.title('({} {})\n Learning Curve'.format(prob, algo))
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.show()

def plot_2(steps_mean, steps_deviations, algo, prob, x_label, y_label, episodes_per_algo):
    x = [i for i in range(1, episodes_per_algo + 1)]
    plt.plot(x, steps_mean, 'k-')
    plt.fill_between(x, np.asarray(steps_mean) - np.asarray(steps_deviations), np.asarray(steps_mean) + np.asarray(steps_deviations))
    plt.title('({} {})\n Learning Curve'.format(prob, algo))
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.show()

run_algo_times = 2
episodes_per_algo = 500
alphas = [0.01]
sigmas = [1]
order = 1

for alpha in alphas:
    overall_actions = []
    overall_actions_before_start = []
    for sigma in sigmas:
        for i in range(run_algo_times):
            actions_count = []
            actions_before_start = [0] * (episodes_per_algo + 1)
            p_wei = get_policy_weights(order)
            v_wei = get_v_weights(order)
            p_fea = getFourierSeries(order)
            v_fea = getFourierSeries(order)
            p_alpha = alpha
            v_alpha = alpha
            for i in range(episodes_per_algo):
                p_wei, v_wei, timesteps = runEpisode(
                    p_wei, v_wei, p_fea, v_fea, p_alpha, v_alpha, sigma)
                actions_count.append(timesteps)
                actions_before_start[i+1] = actions_before_start[i] + timesteps
                print(timesteps)
            overall_actions.append(actions_count)
            overall_actions_before_start.append(actions_before_start)

        avg_overall_actions = np.mean(overall_actions, axis=0)
        std_overall_actions = np.std(overall_actions, axis=0)
        avg_overall_actions_before_start = np.mean(overall_actions_before_start, axis=0)
        
        plot_1(avg_overall_actions_before_start[1:], 'One Step Actor Critic', 'Cart Pole', 'Time steps', 'Episodes', episodes_per_algo)
        plot_2(avg_overall_actions, std_overall_actions, 'One Step Actor Critic', 'Cart Pole', 'Episodes', '#Number of Actions', episodes_per_algo)