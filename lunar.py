import random

import gym
from nn import NN
import math
# from tests import test_calc_next_state_expected_value, test_feed_forward, test_calc_output_grad, test_calc_hidden_grad
from graphing import *
env = gym.make('LunarLander-v2')

np.random.seed(0)
debug = False

epsilon = .01
# dp = []


def choose_action(nn, state):
    action_values = nn.feed_forward(state, False)
    p = np.random.rand()
    if p < epsilon:
        return np.random.randint(0, len(action_values))
    else:
        return np.argmax(action_values)


def update_action_values(nn, state, action, state_prime, reward, done):
    if done:
        target = reward
    else:
        action_values = nn.feed_forward(state_prime, False)
        state_prime_value = calc_state_prime_value(action_values)
        target = reward + state_prime_value

    if debug:
        nn.update_with_cost_check(state, action, target)
    else:
        nn.update_with_cost_check(state, action, target)


def calc_state_prime_value(action_values):
    if len(action_values) == 1:
        print("predicted next state vlaue ",  action_values[0])
        return action_values[0]
    action_weights = np.ones((len(action_values))) * (epsilon / (len(action_values) -1))
    action_weights[np.argmax(action_values)] = 1-epsilon
    prediction = np.dot(action_values, action_weights)

    return prediction


model_size = 2
def update_model():
    global model
    model_entry = {"action": action, "state prime": state_prime, "reward": reward, "state": state, "done": done}
    model.append(model_entry)
    if len(model) > 2:
        model = model[1:]


if __name__ == '__main__':

    avg_reward = 0
    i = 0

    weights = []
    bias = []

    # test_calc_next_state_expected_value()
    # test_feed_forward()
    # test_calc_output_grad()
    # test_calc_hidden_grad()

    if debug:
        layer_node_count = [1,1,1]
        weight_generator = lambda upstream, current : np.ones((upstream, current))
        bias_generator = lambda current : np.ones(current)
        stepper = lambda action: (np.ones(layer_node_count[0]), -1, False, 0)
    else:
        layer_node_count = [8, 32, 4]
        weight_generator = lambda upstream, current : np.random.normal(0, math.sqrt(upstream), (upstream, current))
        bias_generator = lambda current : np.random.normal(0, 1, current)
        stepper = lambda action:  env.step(action)


    upstream_count = layer_node_count[0]

    for i in layer_node_count[1:]:
        weights.append(weight_generator(upstream_count, i))
        bias.append(bias_generator(i))
        upstream_count = i

    config = {
        "weights": weights,
        "bias": bias,
        "learning rate": .00001,
        "debug": debug,
        "batch size": 1
    }

    model = []
    nn = NN(config)
    while True:
        if debug:
            state = np.ones(layer_node_count[0])
        else:
            state = env.reset()

        action = choose_action(nn, state)
        total_reward = 0
        rendering =  i % 100 == 0 and i != 0

        t = 0
        while True:
            if rendering:
                env.render()
            # time.sleep(.001)
            nn.feed_forward(state)
            state_prime, reward, done, info = stepper(action)
            update_model()


            if debug and t % 10 == 0 and t != 0:
                done = True
                reward = 0
                t = 0
            else:
                t +=1

            total_reward += reward
            update_action_values(nn, state, action, state_prime, reward, done)
            if done:
                if reward > 0:
                    print ("done with final reward ", total_reward)
                break
            action = choose_action(nn, state_prime)
            state = state_prime
        exp_replay_size = 2
        if len(model) > exp_replay_size:
            for _ in range(exp_replay_size):
                choice = random.choice(model)
                nn.feed_forward(choice["state"])
                update_action_values(nn, choice["state"], choice["action"], choice["state prime"], choice["reward"], choice["done"])
        avg_reward += total_reward
        i += 1
        rep_freq = 100
        if i % rep_freq == 0:
            graph_average_reward(avg_reward / rep_freq)
            avg_reward = 0
env.close()
