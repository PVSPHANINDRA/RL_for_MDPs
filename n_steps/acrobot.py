import numpy as np
from math import pi, exp, cos
import random
import matplotlib.pyplot as plt
from itertools import product
import gym

discount_parameter = 1
actions = [0, 1, 2]
state_variables = ['c_t1', 's_t1', 'c_t2', 's_t2', 'a_t1', 'a_t2']
state_variable_ranges = [[-1,1],[-1,1],[-1,1],[-1,1],[-12.567, 12.567], [-28.274, 28.274]]


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

def get_q_weights(order):
    total = int(pow(order+1, len(state_variables)))
    return [np.array([0.0] * total).reshape(1, total) for i in actions]

def get_q_eq_output(policy_weights, policy_features, norm_state, cosineFlag=True):
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

def get_q_value(q_weights, q_features, norm_state, cosineFlag=True):
    valSubstitutingInFeatures = np.matmul(q_features, np.array(
        list(norm_state)).reshape(len(norm_state), 1))
    f_q_features = np.cos(
        pi * valSubstitutingInFeatures) if cosineFlag else np.sin(pi * valSubstitutingInFeatures)
    val = np.dot(q_weights, f_q_features)
    return val[(0,0)]

def get_q_feature_value(norm_state, q_features, cosineFlag = True):
    valSubstitutingInFeatures = np.matmul(q_features, np.array(
        list(norm_state)).reshape(len(norm_state), 1))
    f_q_features = np.cos(
        pi * valSubstitutingInFeatures) if cosineFlag else np.sin(pi * valSubstitutingInFeatures)
    return np.transpose(f_q_features)

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


def runEpisode(q_weights, q_features, q_alpha, epsilon, n_steps, cosineFlag=True):
    env = gym.make('Acrobot-v1')
    # initialising the state
    state = env.reset()[0]
    terminated = False

    # normalising the state
    norm_state = getNormValue(state, cosineFlag)
    
    # picking an action
    q_outputs = get_q_eq_output(
    q_weights, q_features, norm_state)
    actionProb = getEpsilonGreedyProbs(q_outputs, epsilon)
    action = random.choices(actions, actionProb)[0] # picked action

    # store
    statesStore = [state]
    actionsStore = [action]
    rewardsStore = []

    max_timesteps = 500
    timesteps = 0

    while True:
        state = statesStore[-1]
        action = actionsStore[-1]
        
        # terminate condition
        if terminated or timesteps == max_timesteps: # terminal state
            return q_weights, timesteps
    
        # gettting next state and its reward. Saving it in store
        next_state, reward, terminated, truncated, info = env.step(action)
        statesStore.append(next_state)
        rewardsStore.append(reward)

        # norm the state
        norm_state = getNormValue(state, cosineFlag)

        # next_state is not a terminal state
        if not terminated:
          # getting the q equation output, using it to find the action by epsilon greedy or softmax
          q_outputs = get_q_eq_output(q_weights, q_features, norm_state)
          actionProb = getEpsilonGreedyProbs(q_outputs, epsilon)
          action = random.choices(actions, actionProb)[0] # picked action
          actionsStore.append(action)
        else:
          max_timesteps = timesteps + 1


        stepper_init_index = timesteps - n_steps + 1

        if stepper_init_index >= 0:
          discount_return = sum([ pow(discount_parameter, i - stepper_init_index - 1) * rewardsStore[i] for i in range(stepper_init_index+1, min(stepper_init_index + n_steps, max_timesteps))])
          if stepper_init_index + n_steps < max_timesteps:
            stepper_last_index = stepper_init_index + n_steps
            next_state = statesStore[stepper_last_index]
            next_action = actionsStore[stepper_last_index]
            next_actionIndex = actions.index(next_action)

            norm_next_state = getNormValue(next_state)
            discount_return += pow(discount_parameter, n_steps) * get_q_value(q_weights[next_actionIndex], q_features, norm_next_state, cosineFlag)

          stepper_init_state = statesStore[stepper_init_index]
          stepper_init_action = actionsStore[stepper_init_index]
          stepper_init_actionIndex = actions.index(stepper_init_action)
          norm_stepper_init_state = getNormValue(stepper_init_state)

          # updating weights
          for i in range(len(actions)):
            q_weights[i] += float(q_alpha) * (discount_return - get_q_value(q_weights[stepper_init_actionIndex], q_features, norm_stepper_init_state, cosineFlag)) * get_q_feature_value(norm_stepper_init_state, q_features, cosineFlag)

        timesteps += 1

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


run_algo_times = 20
episodes_per_algo = 2000
alphas = [0.001]
epsilons = [0.1]
n_steps = 8
order = 1
decay_epsilon_param = 0

for alpha in alphas:
    overall_actions = []
    overall_actions_before_start = []
    for epsilon in epsilons:
        for i in range(run_algo_times):
            decay_epsilon = epsilon
            actions_count = []
            actions_before_start = [0] * (episodes_per_algo + 1)
            q_wei = get_q_weights(order)
            q_fea = getFourierSeries(order)
            q_alpha = alpha
            for i in range(episodes_per_algo):
                if (i+1) % 50 == 0 and decay_epsilon > 0:
                    decay_epsilon -= decay_epsilon_param
                    decay_epsilon = max(0, decay_epsilon)
                q_wei, timesteps = runEpisode(
                    q_wei, q_fea, q_alpha, decay_epsilon, n_steps, True)
                actions_count.append(timesteps)
                actions_before_start[i+1] = actions_before_start[i] + timesteps
                print(timesteps)
            overall_actions.append(actions_count)
            overall_actions_before_start.append(actions_before_start)

        avg_overall_actions = np.mean(overall_actions, axis=0)
        std_overall_actions = np.std(overall_actions, axis=0)
        avg_overall_actions_before_start = np.mean(overall_actions_before_start, axis=0)

        plot_1(avg_overall_actions_before_start[1:], 'n-steps Semi Gradient', 'Acrobot', 'Time steps', 'Episodes', episodes_per_algo)
        plot_2(avg_overall_actions, std_overall_actions, 'n-steps Semi Gradient', 'Acrobot', 'Episodes', '#Number of Actions', episodes_per_algo)