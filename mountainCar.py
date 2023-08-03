import math
import sys

mountain_car_x_range = [-1.2, 0.5]
mountain_car_v_range = [-0.07, 0.07]

mountain_car_terminal_x = 0.5

state_variables_for_mountain_car = ['x', 'v']

# Best performance for no_of_x_divisions, no_of_v_divisions = 20, 10
try:
    no_of_x_divisions = int(sys.argv[2])
except Exception:
    no_of_x_divisions = 20
try:
    no_of_v_divisions = int(sys.argv[3])
except Exception:
    no_of_v_divisions = 10


def round_off(n):
    return round(n, 6)


mountain_car_x_step = round_off((mountain_car_x_range[1] - mountain_car_x_range[0]) / no_of_x_divisions)
mountain_car_v_step = round_off((mountain_car_v_range[1] - mountain_car_v_range[0]) / no_of_v_divisions)


def formulate_the_states():
    states = []
    x_value = mountain_car_x_range[0]
    while x_value < mountain_car_x_range[1]:
        x_lower_bound = x_value
        x_value = min(mountain_car_x_range[1], round_off(x_value + mountain_car_x_step))
        x_upper_bound = x_value
        v_value = mountain_car_v_range[0]
        while v_value < mountain_car_v_range[1]:
            v_lower_bound = v_value
            v_value = min(mountain_car_v_range[1], round_off(v_value + mountain_car_v_step))
            v_upper_bound = v_value
            states.append(((x_lower_bound, x_upper_bound), (v_lower_bound, v_upper_bound)))
    states.sort()
    return states


def fetch_terminal_states_for_mountain_car(states):
    t_states = []
    for s in states:
        if s[0][1] == mountain_car_terminal_x:
            t_states.append(s)
    return t_states


mountain_car_states = formulate_the_states()
# mountain_car_terminal_states = fetch_terminal_states_for_mountain_car(mountain_car_states)
