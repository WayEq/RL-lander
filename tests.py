# import numpy as np
#
# from nn import NN
#
#
# def test_calc_next_state_expected_value():
#     output = calc_state_prime_value([1, 2, 3, 4])
#     # if output != 4 * (1 - epsilon) + 1 * epsilon/3 + 2 * epsilon/3 + 3 * epsilon/3:
#     #     raise ValueError("expected next state value is wrong")
#
#
# def test_feed_forward():
#     test_num_input = 1
#     test_num_hidden = 2
#     test_num_output = 1
#
#     test_hidden_weights = np.ones((test_num_input, test_num_hidden))
#     test_out_weights = np.ones((test_num_hidden, test_num_output)) * 2
#     test_hidden_bias = np.ones(test_num_hidden)
#     test_out_bias = np.ones(test_num_output)
#     nn = NN(test_num_input, test_num_hidden, test_num_output, test_hidden_weights, test_hidden_bias, test_out_weights, test_out_bias)
#
#     state = np.ones(test_num_input) * 10
#     nn.feed_forward(state)
#     if not np.allclose(nn.last_hidden_output, np.ones(test_num_hidden) * test_num_input * 10 + 1):
#         raise ValueError("hidden outputs are off")
#     if not np.allclose(nn.last_output, np.sum(nn.last_hidden_output* 2) + 1):
#         raise ValueError("output outputs are off")
#
#
# def test_calc_output_grad():
#     test_num_input = 1
#     test_num_hidden = 2
#     test_num_output = 1
#
#     test_hidden_weights = np.ones((test_num_input, test_num_hidden))
#     test_out_weights = np.ones((test_num_hidden, test_num_output))
#     test_hidden_bias = np.zeros(test_num_hidden)
#     test_out_bias = np.zeros(test_num_output)
#     nn = NN(test_num_input, test_num_hidden, test_num_output, test_hidden_weights, test_hidden_bias, test_out_weights, test_out_bias)
#     nn.feed_forward([1])
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