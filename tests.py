import numpy as np
#
from nn import NN


#
#
# def test_calc_next_state_expected_value():
#     output = calc_state_prime_value([1, 2, 3, 4])
#     # if output != 4 * (1 - epsilon) + 1 * epsilon/3 + 2 * epsilon/3 + 3 * epsilon/3:
#     #     raise ValueError("expected next state value is wrong")
#
#
def run_tests():
    test_feed_forward()
    test_calc_output_grad()


def test_feed_forward():
    config = {
        "learning rate": .0001,
        "debug": False,
        "layer node count": [2, 2, 2],
        "first moment decay rate": .9,
        "second moment decay rate": .999,
        "experience replay size": 8,
        "replay batches": 4,
        "model size": 50000,
        "report frequency": 10,
        "number episodes": 300000000,
        "initial temp": 1,
        "adam epsilon": 10 ** -8
    }
    num_inputs = config["layer node count"][0]
    num_hidden = config["layer node count"][1]
    num_output = config["layer node count"][2]
    config["weights"] = [np.ones((num_inputs, num_hidden)), np.ones((num_hidden, num_output)) * 2]
    config["bias"] = [np.ones((num_hidden)), np.ones((num_output)) * 2]

    nn = NN(config)

    state = np.ones(num_inputs) * 1
    state2 = np.ones(num_inputs) * 2
    state3 = np.ones(num_inputs) * 3
    states = np.row_stack((state, state2, state3))
    zs, activations = nn.feed_forward(states)
    for i in range(3):
        if not np.allclose(activations[-2][i], num_inputs * (i + 1) + 1):
            raise ValueError("hidden outputs are off")
    for j in range(3):
        if not np.allclose(activations[-1][j], ((num_inputs * (j + 1) + 1) * num_hidden * 2) + 2):
            raise ValueError("hidden outputs are off")

    # Test RELU

    state = np.ones(num_inputs) * -2
    states = state.reshape((1, 2))
    zs, activations = nn.feed_forward(states)
    if not np.allclose(activations[-2][:], 0):
        raise ValueError("hidden outputs are off")

    if not np.allclose(activations[-1][:], 2):  # only bias..
        raise ValueError("hidden outputs are off")


def test_calc_output_grad():
    config = {
        "learning rate": .0001,
        "debug": False,
        "layer node count": [2, 2, 2],
        "first moment decay rate": .9,
        "second moment decay rate": .999,
        "experience replay size": 8,
        "replay batches": 4,
        "model size": 50000,
        "report frequency": 10,
        "number episodes": 300000000,
        "initial temp": 1,
        "adam epsilon": 10 ** -8
    }

    num_inputs = config["layer node count"][0]
    num_hidden = config["layer node count"][1]
    num_output = config["layer node count"][2]
    config["weights"] = [np.ones((num_inputs, num_hidden)), np.ones((num_hidden, num_output))]
    config["bias"] = [np.ones((num_hidden)), np.ones((num_output))]
    states = []
    action_indices = []
    targets = []
    finals = []
    for i in range(2):
        state = np.array([i, i])
        action_index = i
        target = i
        states.append(state)
        action_indices.append(action_index)
        targets.append(target)
        finals.append(i == 1)

    nn = NN(config)
    bg, wg = nn.calc_gradients(states, action_indices, states, np.ones(2), finals, 1, 1)
    test_weight_g = [np.array([[3., 3.],
                                          [3., 3.]]), np.array([[-.5, 9.],
                                                                [-.5, 9.]])]
    test_bias_g = [np.array([[2.5, 2.5]]), np.array([[-.5, 3.]])]

    for i in range(2):
        if not np.allclose(wg[i], test_weight_g[i]):
            raise ValueError("Error in calc weight grad")

        if not np.allclose(bg[i], test_bias_g[i]):
            raise ValueError("Error in calc weight grad")
#     error1 = nn.last_output[0] - 0
#     output_weights_gradient = nn.last_hidden_output * error1
#     output_bias_gradient = error1
#     result = error1, output_bias_gradient, output_weights_gradient
#     error, out_bias_gradient, out_weights_gradient = result
#     if not np.allclose(2, out_weights_gradient):
#         raise ValueError("out weight gradient is off")
#
#     if not np.allclose(2, out_bias_gradient):
#         raise ValueError("out bias gradient is off")
#
#
# def test_calc_hidden_grad():
#     test_num_input = 1
#     test_num_hidden = 2
#     test_num_output = 1
#
#     test_hidden_weights = np.ones((test_num_output, test_num_hidden))
#     test_out_weights = np.ones((test_num_hidden, test_num_output))
#     test_hidden_bias = np.zeros(test_num_hidden)
#     test_out_bias = np.zeros(test_num_output)
#     nn = NN(test_num_input, test_num_hidden, test_num_output, test_hidden_weights, test_hidden_bias, test_out_weights, test_out_bias)
#     nn.feed_forward([1])
#     hidden_error, hidden_weights_gradient, hidden_bias_gradient = nn.calc_hidden_nodes_gradient(2,0)
#     if not np.allclose(2, hidden_weights_gradient):
#         raise ValueError("hidden weight gradient is off")
#
#     if not np.allclose(2, hidden_bias_gradient):
#         raise ValueError("hidden bias gradient is off")
