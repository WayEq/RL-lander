import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
env = gym.make('LunarLander-v2')

np.random.seed(0)
debug = True
debug = False

mapper = lambda t: 0 if t < 0 else 1
f = np.vectorize(mapper)
epsilon = .01
step_size = .0001
dp = []
class NN(object):

    def __init__(self, num_input, num_hidden, num_output, weights_hidden, bias_hidden, weights_output, bias_output):
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.weights_hidden = weights_hidden
        self.bias_hidden = bias_hidden
        self.weights_output = weights_output
        self.bias_output = bias_output
        self.last_hidden_output = 0
        self.hidden_z = 0
        self.last_output = []
        self.last_input = 0

    def feed_forward(self, state):
        self.last_input = state
        self.hidden_z, self.last_hidden_output, self.last_output = self.get_layer_outputs_for_state(state)

    def get_layer_outputs_for_state(self, state):
        dot_prod = np.dot(state, self.weights_hidden) + self.bias_hidden
        hidden_z = dot_prod
        hidden_output = np.maximum(dot_prod, 0)
        output = np.dot(hidden_output, self.weights_output) + self.bias_output
        return np.copy(hidden_z), np.copy(hidden_output), np.copy(output)

    def update(self, action_node_index, target):
        pre_cost = ((target - self.last_output[action_node_index]) ** 2) / 2.0

        output_error, output_bias_gradient, output_weights_gradient = self.calc_output_gradients(action_node_index, target)
        #
        # old_weights_out = np.copy(self.weights_output)
        # old_bias_out =  np.copy(self.bias_output)

        self.weights_output[:, action_node_index] -= output_weights_gradient * step_size
        self.bias_output[action_node_index] -= output_bias_gradient * step_size


        hidden_error, hidden_weights_gradient, hidden_bias_gradient = self.calc_hidden_nodes_gradient(output_error, action_node_index)

        if np.random.rand() > .99:
            self.debug_weights()

        # print("output bias grad ", output_bias_gradient)

        # old_hidden_weights, old_hidden_bias  = np.copy(self.weights_hidden), np.copy(self.bias_hidden)
        self.weights_hidden -= hidden_weights_gradient * step_size
        self.bias_hidden -= hidden_bias_gradient * step_size

        # new_cost = ((target - self.get_layer_outputs_for_state(state)[2][action_node_index]) ** 2) / 2.0
        # cost_change = new_cost - pre_cost
        # print("c_delta", cost_change)
        # global dp
        #dp.append(cost_change)

        # if (cost_change > 0):
        #     s = pd.Series(dp[-1000:])
        #     s.plot()
        #     plt.show()
        #     raise ValueError("Cost increased from ", pre_cost, " to ", new_cost)



    def calc_output_gradients(self, action_node_index, target):
        error = self.last_output[action_node_index] - target
        output_weights_gradient = self.last_hidden_output * error
        output_bias_gradient = error
        return error, output_bias_gradient, output_weights_gradient


    def calc_hidden_nodes_gradient(self, output_error, action):
        activation_derivative = f(self.hidden_z)

        error = self.weights_output[:,action] * output_error * activation_derivative

        #print("error (hidden)", error)
        tile = np.tile(self.last_input, (self.num_hidden, 1))
        hidden_weights_gradient = np.transpose(tile) * error
        hidden_bias_gradient = error
        return error, hidden_weights_gradient, hidden_bias_gradient

    def debug_weights(self):
        array_sum = np.sum(self.weights_hidden)
        array_has_nan = np.isnan(array_sum)
        if array_has_nan:
            raise ValueError("nan in hidden")
        array_sum = np.sum(self.weights_output)
        array_has_nan = np.isnan(array_sum)
        if array_has_nan:
            raise ValueError("nan in output")



def choose_action(nn, state):
    p = np.random.rand()
    if p < epsilon:
        return np.random.randint(0, num_actions)
    else:
        outputs = nn.get_layer_outputs_for_state(state)[2]
        return np.argmax(outputs)


def update_action_values(nn, action, state_prime, reward, done, state):
    if done:
        target = reward
    else:
        action_values = nn.get_layer_outputs_for_state(state_prime)[2]
        state_prime_value = calc_state_prime_value(action_values)
        target = reward + state_prime_value

    nn.update(action, target)


def calc_state_prime_value(action_values):
    if (len(action_values) == 1):
        print("predicted next state vlaue ",  action_values[0])
        return action_values[0]
    weights = np.ones((len(action_values))) * (epsilon / (len(action_values) -1))
    weights[np.argmax(action_values)] = 1-epsilon
    prediction = np.dot(action_values, weights)
    # print("predicted next state vlaue ", prediction)
    # if prediction < -16.28:
    #    print("shit")
    return prediction


# TEST
def test_calc_next_state_expected_value():
    output = calc_state_prime_value([1, 2, 3, 4])
    if output != 4 * (1 - epsilon) + 1 * epsilon/3 + 2 * epsilon/3 + 3 * epsilon/3:
        raise ValueError("expected next state value is wrong")


def test_feed_forward():
    test_num_input = 1
    test_num_hidden = 2
    test_num_output = 1

    test_hidden_weights = np.ones((test_num_input, test_num_hidden))
    test_out_weights = np.ones((test_num_hidden, test_num_output)) * 2
    test_hidden_bias = np.ones(test_num_hidden)
    test_out_bias = np.ones(test_num_output)
    nn = NN(test_num_input, test_num_hidden, test_num_output, test_hidden_weights, test_hidden_bias, test_out_weights, test_out_bias)

    state = np.ones(test_num_input) * 10
    nn.feed_forward(state)
    if not np.allclose(nn.last_hidden_output, np.ones(test_num_hidden) * test_num_input * 10 + 1):
        raise ValueError("hidden outputs are off")
    if not np.allclose(nn.last_output, np.sum(nn.last_hidden_output* 2) + 1):
        raise ValueError("output outputs are off")


def test_calc_output_grad():
    test_num_input = 1
    test_num_hidden = 2
    test_num_output = 1

    test_hidden_weights = np.ones((test_num_input, test_num_hidden))
    test_out_weights = np.ones((test_num_hidden, test_num_output))
    test_hidden_bias = np.zeros(test_num_hidden)
    test_out_bias = np.zeros(test_num_output)
    nn = NN(test_num_input, test_num_hidden, test_num_output, test_hidden_weights, test_hidden_bias, test_out_weights, test_out_bias)
    nn.feed_forward([1])
    error, out_bias_gradient, out_weights_gradient = nn.calc_output_gradients(0, 0)
    if not np.allclose(2, out_weights_gradient):
        raise ValueError("out weight gradient is off")

    if not np.allclose(2, out_bias_gradient):
        raise ValueError("out bias gradient is off")


def test_calc_hidden_grad():
    test_num_input = 1
    test_num_hidden = 2
    test_num_output = 1

    test_hidden_weights = np.ones((test_num_output, test_num_hidden))
    test_out_weights = np.ones((test_num_hidden, test_num_output))
    test_hidden_bias = np.zeros(test_num_hidden)
    test_out_bias = np.zeros(test_num_output)
    nn = NN(test_num_input, test_num_hidden, test_num_output, test_hidden_weights, test_hidden_bias, test_out_weights, test_out_bias)
    nn.feed_forward([1])
    hidden_error, hidden_weights_gradient, hidden_bias_gradient = nn.calc_hidden_nodes_gradient(2,0)
    if not np.allclose(2, hidden_weights_gradient):
        raise ValueError("hidden weight gradient is off")

    if not np.allclose(2, hidden_bias_gradient):
        raise ValueError("hidden bias gradient is off")



# END TEST

if __name__ == '__main__':

    avg_reward = 0
    i = 0

    if debug:
        num_actions = 1
        num_input = 1
        num_hidden = 1
        # init_weights_hidden = np.ones((num_input, num_hidden))
        # init_bias_hidden = np.ones((num_hidden))
        # init_weights_out = np.ones((num_hidden, num_actions))
        # init_bias_out = np.ones((num_actions))
        init_weights_hidden = np.random.rand(num_input, num_hidden)
        init_bias_hidden = np.random.rand(num_hidden)
        init_weights_out = np.random.rand(num_hidden, num_actions)
        init_bias_out = np.random.rand(num_actions)
    else:
        test_calc_next_state_expected_value()
        test_feed_forward()
        test_calc_output_grad()
        test_calc_hidden_grad()
        num_actions = 4
        num_input = 8
        num_hidden = 16
        init_weights_hidden = np.random.rand(num_input, num_hidden)
        init_bias_hidden = np.random.rand(num_hidden)
        init_weights_out = np.random.rand(num_hidden, num_actions)
        init_bias_out = np.random.rand(num_actions)

    nn = NN(num_input, num_hidden, num_actions, init_weights_hidden, init_bias_hidden, init_weights_out, init_bias_out)
    while True:
        if debug:
            state = np.ones(num_input)
        else:
            state = env.reset()
        action = choose_action(nn, state)
        total_reward = 0
        rendering = i % 50 == 0 and i != 0
        for t in range(10000):
            if rendering:
                env.render()
            # time.sleep(.001)
            nn.feed_forward(state)
            if debug:
                state_prime, reward, done = np.ones(num_input), -1, False
                if t != 0 and t % 10 == 0:
                    done = True
                    reward = 0
            else:
                state_prime, reward, done, info = env.step(action)

            total_reward += reward
            update_action_values(nn, action, state_prime, reward, done, state)
            if done:
                break
            action = choose_action(nn, state_prime)
            state = state_prime

        avg_reward += total_reward
        i += 1
        rep_freq = 50
        if i % rep_freq == 0:
            print("Avg reward now ", avg_reward / rep_freq)
            avg_reward = 0
env.close()
