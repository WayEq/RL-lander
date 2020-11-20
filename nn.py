import numpy as np

from graphing import live_plotter

mapper = lambda t: 0 if t < 0 else 1

f = np.vectorize(mapper)

class NN(object):

    def debug_weights(self):
        array_sum = np.sum(self.weights[0])
        array_has_nan = np.isnan(array_sum)
        if array_has_nan:
            print(self.weights)
            print(self.bias)
            raise ValueError("nan in hidden")
        array_sum = np.sum(self.weights[1])
        array_has_nan = np.isnan(array_sum)
        if array_has_nan:
            print(self.weights)
            print(self.bias)
            raise ValueError("nan in output")

    def __init__(self, config):
        self.config = config
        self.weights = config["weights"]
        self.bias = config["bias"]
        self.zs = []
        self.activations = []
        self.num_layers = len(self.bias)
        self.batch_count = 0
        self.batch_weight_grad = []
        self.batch_bias_grad = []


        # Debug stuff
        self.debug_hidden_bias = []
        self.debug_hidden_weights = []
        self.debug_out_bias = []
        self.debug_out_weights = []
        self.debug_out_weights_line = []

    def feed_forward(self, state, update=True):
        prev_activations = state
        zs = []
        activations = []
        for i in range(self.num_layers):
            zs.append(np.dot(prev_activations, self.weights[i]) + self.bias[i])
            activations.append(zs[i])
            if i != self.num_layers -1:
                # RELU
                activations[i] = np.maximum(0,activations[i])
            prev_activations = activations[i]
        if update:
            self.zs = zs
            self.activations = activations
        return activations[-1]

    def update_with_cost_check(self, input, action_node_index, target):
        self.update(input,action_node_index, target)

    def update(self, input, action_node_index, target):
        bias_grad, weight_grad = self.calc_gradients(action_node_index, input, target)
        self.debug_cost(input, target, action_node_index, bias_grad, weight_grad)
        if len(self.batch_bias_grad) == 0:
            self.batch_bias_grad = bias_grad
            self.batch_weight_grad = weight_grad
        else:
            for i in range(self.num_layers):
                self.batch_bias_grad[i] += bias_grad[i]
                self.batch_weight_grad[i] += weight_grad[i]
        self.batch_count += 1
        if self.batch_count == self.config["batch size"]:
            for i in range(self.num_layers):
                learning_rate = self.config["learning rate"]
                self.weights[i] -= self.batch_weight_grad[i] * learning_rate / self.config["batch size"]
                self.bias[i] -= self.batch_bias_grad[i] * learning_rate / self.config["batch size"]

            # self.debug_out_weights.append(np.copy(self.weights[-1][0]))
            # self.debug_out_bias.append(self.bias[-1])

            if np.random.rand() > 0:
                self.debug_weights()
            self.batch_bias_grad = []
            self.batch_weight_grad = []
            self.batch_count = 0

    def calc_gradients(self, action_node_index, input, target):
        weight_grad = []
        bias_grad = []
        # Last layer
        masked = np.copy(self.activations[-1])
        masked[action_node_index] = target
        error = self.activations[-1] - masked
        weight_grad.append(self.activations[-2][:, np.newaxis] * error)
        bias_grad.append(error)
        for i in reversed(range(self.num_layers - 1)):
            error = np.dot(error, self.weights[i + 1].transpose())
            upstream = input
            if i != 0:
                upstream = self.activations[i - 1]
            activation_derivative = [ i.__float__() for i in (self.zs[i] > 0)]

            # dw_dz = upstream * activation_derivative
            weight_grad.insert(0, upstream[:, np.newaxis] * (error * activation_derivative))
            bias_grad.insert(0, error)
        return bias_grad, weight_grad

    def debug_cost(self, input, target, action_node_index, bias_grad, weight_grad):

        pre_cost = ((target - self.activations[-1][action_node_index]) ** 2) / 2.0
        pre_weights = [ np.copy(i) for i in self.weights]
        pre_bias = [ np.copy(i) for i in self.bias]
        for i in range(self.num_layers):
            learning_rate = self.config["learning rate"]
            self.weights[i] -= weight_grad[i] * learning_rate
            self.bias[i] -= bias_grad[i] * learning_rate

        new_cost = ((target - self.feed_forward(input, False)[action_node_index]) ** 2) / 2.0
        cost_change = new_cost - pre_cost
        # print("c_delta", cost_change)
        # global dp
        # dp.append(cost_change)
        if cost_change > 0:
            # s = pd.Series(dp)
            # s.plot()
            # plt.show()
            print("Cost increased from ", pre_cost, " to ", new_cost)
            if self.config["debug"]:
                self.graph_params()
            raise ValueError("cost")
        self.weights = pre_weights
        self.bias = pre_bias
    def graph_params(self):
        self.debug_out_weights_line = live_plotter(range(len(self.debug_out_weights)), self.debug_out_weights, self.debug_out_weights_line)