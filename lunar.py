import gym
import tests
from nn import NN

from graphing import *

np.random.seed(1)


def init_saxe(rows, cols):

    tensor = np.random.normal(0, 1, (rows, cols))
    if rows < cols:
        tensor = tensor.T
    tensor, r = np.linalg.qr(tensor)
    d = np.diag(r, 0)
    ph = np.sign(d)
    tensor *= ph

    if rows < cols:
        tensor = tensor.T
    return tensor


def initialize():
    env = gym.make('LunarLander-v2')
    # env.seed(0)
    config = {
        "learning rate": 1e-3,
        "debug": False,
        "layer node count": [8, 256, 4],
        "first moment decay rate": .9,
        #"first moment decay rate": 0,
        "second moment decay rate": .999,
        #"second moment decay rate": 0,
        "experience replay size": 8,
        "replay batches": 4,
        "model size": 50000,
        "discount": .99,
        "report frequency": 10,
        "number episodes": np.inf,
        "initial temp": .001,
        "temp adjustment": 1,
        "adam epsilon": 1e-8
    }

    if config["debug"]:
        config["layer node count"] = [2, 2, 1]

        def weight_generator(upstream, current):
            return np.ones((upstream, current))

        def bias_generator(current):
            return np.ones(current)

        def stepper(_):
            input_count = config["layer node count"][0]
            return np.ones(input_count), -1, False, 0

        def init_env():
            return np.ones(config["layer node count"][0])
    else:
        # layer_node_count = [8, 8, 4]

        def weight_generator(upstream, current):
            return init_saxe(upstream,current)
            #return np.random.normal(0, 1, (upstream, current)) / np.sqrt(upstream)

        def bias_generator(current):
            return np.zeros(current)

        def stepper(action):
            return env.step(action)

        def init_env():
            return env.reset()

    weights = []
    bias = []

    upstream_count = config["layer node count"][0]
    for i in config["layer node count"][1:]:
        weights.append(weight_generator(upstream_count, i))
        bias.append(bias_generator(i))
        upstream_count = i

    config["weights"] = weights
    config["bias"] = bias
    return config, stepper, init_env, env


def choose_action(nn, state, temp):
    action_values = nn.feed_forward(state)[1][-1]
    action_probabilities = nn.get_probabilities(action_values[np.newaxis, :], temp)
    choice_range = len(action_values)
    choice = np.random.choice(choice_range, p=action_probabilities[0])
    predicted_value = np.dot(action_values, action_probabilities[0])
    return choice, predicted_value


def update_model(config, model, state, action, state_prime, reward, done):
    model_entry = [state, action, state_prime, reward, done]
    model.append(model_entry)
    if len(model) > config["model size"]:
        del model[0]


def main():
    config, stepper, init_env, env = initialize()

    for learning_rate in [config["learning rate"]]:
        print("using learning rate ", learning_rate)

        config, stepper, init_env, env = initialize()
        debug = config["debug"]
        model = []
        nn = NN(config)
        rendering = False
        total_episode_count = 0

        temp = config["initial temp"]
        while total_episode_count < config["number episodes"]:
            batch_episode_count = 0
            batch_reward = 0
            while batch_episode_count < config["report frequency"]:
                batch_reward += run_episode(config, debug, env,
                                            init_env, model, nn, rendering, stepper, temp,
                                            )
                batch_episode_count += 1
                temp = temp * config["temp adjustment"]
                rendering = False
            total_episode_count += batch_episode_count
            add_value(batch_reward / batch_episode_count, "Avg reward")
            live_plotter()
            rendering = True
            print("avg reward this batch: ", batch_reward / batch_episode_count)
        env.close()


def run_episode(config, debug, env, init_env, model, nn, rendering, stepper, temp):
    episode_total_reward = 0
    episode_time_step = 0

    state = init_env()
    action_selected, prediction = choose_action(nn, state, temp)

    episode_prediction = prediction
    while True:

        if not debug and rendering:
            env.render()

        state_prime, reward, done, info = stepper(action_selected)

        if done and reward == 100:
            print("plus 100!  action", action_selected, "\nnew state", state_prime, "\nreward ", reward)

        if debug and episode_time_step == 9:
            done = True
            reward = 0

        update_model(config, model, state, action_selected, state_prime, reward, done)
        episode_total_reward += reward
        # update_action_values(nn, [[state, action, state_prime, reward, done]])
        if len(model) > config["experience replay size"]:
            for i in range(config["replay batches"]):
                choices = np.random.choice(len(model), min(len(model), config["experience replay size"]))
                states = [model[i][0] for i in choices]
                actions = [model[i][1] for i in choices]
                next_states = [model[i][2] for i in choices]
                rewards = [model[i][3] for i in choices]
                finals = [model[i][4] for i in choices]
                nn.adam_update(states, actions, next_states, rewards, finals, temp, config["discount"])
        if done:
            print("done with final reward ", reward, " total ", episode_total_reward, " temp ", temp, " in ", episode_time_step)
            break
        action_selected, prediction = choose_action(nn, state_prime, temp)
        if state_prime[6] == 1.0 and state_prime[7] == 1.0 and state_prime[3] < .02 and state_prime[2] < .02:
            print("touchdown", state, " reward ", reward, " next action ", action_selected)
            #action_selected = 0
        episode_prediction += prediction
        state = state_prime
        episode_time_step += 1
    delta = episode_total_reward - (episode_prediction / episode_time_step)
    add_value(delta, "prediction delta ")
    return episode_total_reward


tests.run_tests()
main()
