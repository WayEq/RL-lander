import numpy as np

from graphing import *
import copy

np.random.seed(1)


class NN(object):

    def __init__(self, config):
        self.config = config
        self.weights = config["weights"]
        self.bias = config["bias"]
        self.adam_m_modifier = self.config["first moment decay rate"]
        self.adam_variance_modifier = self.config["second moment decay rate"]
        self.adam_epsilon = config["adam epsilon"]
        self.learning_rate = self.config["learning rate"]

        self.num_layers = len(self.bias)

        # ADAM
        self.weights_first_moment = [np.zeros(np.shape(i)) for i in self.weights]
        self.bias_first_moment = [np.zeros(np.shape(i)) for i in self.bias]

        self.weights_second_moment = [np.zeros(np.shape(i)) for i in self.weights]
        self.bias_second_moment = [np.zeros(np.shape(i)) for i in self.bias]

    def feed_forward(self, states):
        prev_activations = states
        zs = []
        activations = []
        for i in range(self.num_layers):
            zs.append(np.dot(prev_activations, self.weights[i]) + self.bias[i])
            if i != self.num_layers - 1:
                # RELU
                activations.append(np.maximum(0, zs[i]))
            else:
                activations.append(copy.deepcopy(zs[i]))
            prev_activations = activations[i]

        return zs, activations

    def get_probabilities(self, action_values, temp):
        temp_adjusted_action_values = action_values / temp
        max_value = np.max(temp_adjusted_action_values, axis=1)
        weights = np.exp(temp_adjusted_action_values - max_value[:, np.newaxis])
        sum_total = np.sum(weights, axis=1)
        action_probabilities = weights / np.sum(weights, axis=1)[:, np.newaxis]
        if np.random.random() > .9999:
            print("action values: ", action_values)
        return action_probabilities

    def get_cost(self, states, action_indices, targets):
        costs = np.zeros((len(states)), dtype=float)
        for i in range(len(states)):
            costs[i] = ((1 / 2) * (targets[i] - self.feed_forward(states[i])[1][-1][action_indices[i]]) ** 2)
        return costs

    def get_next_states_prediction(self, next_states, finals, temp):
        activations = self.feed_forward(next_states)[1][-1]

        action_probabilities = self.get_probabilities(activations, temp)
        state_prime_values = np.sum(np.multiply(activations, action_probabilities), axis=1) * (1 - np.array(finals))
        return state_prime_values

    def adam_update(self, states, actions, next_states, rewards, finals, temp, discount):
        # pre_costs = self.get_cost(states, actions, targets)
        batch_bias_grad, batch_weight_grad = self.calc_gradients(states, actions, next_states, rewards, finals, temp,
                                                                 discount)

        first_moment_decay_rate = self.config["first moment decay rate"]
        second_moment_decay_rate = self.config["second moment decay rate"]
        inverse_first_moment_decay_rate = 1 - first_moment_decay_rate
        inverse_second_moment_decay_rate = 1 - second_moment_decay_rate
        unbiased_weights_first_moment = []
        unbiased_bias_first_moment = []
        unbiased_weights_second_moment = []
        unbiased_bias_second_moment = []
        for i in range(self.num_layers):
            self.weights_first_moment[i] = self.weights_first_moment[
                                               i] * first_moment_decay_rate + inverse_first_moment_decay_rate * \
                                           batch_weight_grad[i]
            self.bias_first_moment[i] = self.bias_first_moment[
                                            i] * first_moment_decay_rate + inverse_first_moment_decay_rate * \
                                        batch_bias_grad[i]

            bias_corrector = (1 - self.adam_m_modifier)
            unbiased_weights_first_moment.append(self.weights_first_moment[i] / bias_corrector)
            unbiased_bias_first_moment.append(self.bias_first_moment[i] / bias_corrector)

            # 2nd moment
            self.weights_second_moment[i] = self.weights_second_moment[
                                                i] * second_moment_decay_rate + inverse_second_moment_decay_rate * np.square(
                batch_weight_grad[i])
            self.bias_second_moment[i] = self.bias_second_moment[
                                             i] * second_moment_decay_rate + inverse_second_moment_decay_rate * np.square(
                batch_bias_grad[i])
            second_moment_bias_corrector = (1 - self.adam_variance_modifier)
            unbiased_weights_second_moment.append(self.weights_second_moment[i] / second_moment_bias_corrector)
            unbiased_bias_second_moment.append(self.bias_second_moment[i] / second_moment_bias_corrector)

        for i in range(self.num_layers):
            weight_denominator = np.sqrt(unbiased_weights_second_moment[i]) + self.adam_epsilon
            weights_gradient = unbiased_weights_first_moment[i] / weight_denominator
            weight_delta = self.learning_rate * weights_gradient
            self.weights[i] -= weight_delta
            bias_denominator = np.sqrt(unbiased_bias_second_moment[i]) + self.adam_epsilon
            bias_gradient = unbiased_bias_first_moment[i] / bias_denominator
            bias_delta = self.learning_rate * bias_gradient[0]
            self.bias[i] -= bias_delta
            # print("wd ", weight_delta, " bd ", bias_delta)

        # print(self.adam_m_modifier)
        self.adam_m_modifier *= first_moment_decay_rate
        self.adam_variance_modifier *= second_moment_decay_rate

        if np.random.rand() > .999:
            self.debug_weights()
        # post_costs = self.get_cost(states, action_indices)
        # cost_delta = np.sum(post_costs - pre_costs)

        return 0  # post_costs - pre_costs

    def calc_gradients(self, states, actions, next_states, rewards, finals, temp, discount):
        weight_grad = []
        bias_grad = []
        # Last layer
        states_stack = np.row_stack(states)
        zs, activations = self.feed_forward(states_stack)

        error = self.get_td_error(activations, actions, next_states, rewards, finals, temp, discount)
        last_hidden_activations = activations[-2][np.newaxis, :]
        rotated_z_act = np.transpose(last_hidden_activations, axes=(1, 2, 0))
        rotated_z_err = np.transpose(error[np.newaxis,:], axes=(1, 0, 2))
        weight_grad.append(rotated_z_act @ rotated_z_err)
        bias_grad.append(rotated_z_err)
        for i in reversed(range(self.num_layers - 1)):
            rotated_z_err = rotated_z_err @ self.weights[i + 1].T
            upstream = states_stack
            if i != 0:
                upstream = activations[i - 1]
            np_abs = np.abs(zs[i])
            activation_derivative = (zs[i] + np_abs) / (2 * np_abs)
            z_a_d = activation_derivative[np.newaxis, :]
            rotated_act_derivitive = np.transpose(z_a_d, axes=(1, 0, 2))

            rotated_z_act = np.transpose(upstream[np.newaxis, :], axes=(1, 2, 0))
            rotated_z_err = rotated_z_err * rotated_act_derivitive
            weight_grad.insert(0, rotated_z_act @ rotated_z_err)
            bias_grad.insert(0, rotated_z_err)
        # final_delta = target - activations[-1][action_node_index]
        # add_value(final_delta, "prediction error")
        # self.debug_magnitudes(bias_grad, weight_grad)
        bias_grad = [np.average(i, axis=0) for i in bias_grad]
        weight_grad = [np.average(i, axis=0) for i in weight_grad]
        return bias_grad, weight_grad

    def get_td_error(self, activations, actions, next_states, rewards, finals, temp, discount):
        next_states_prediction = self.get_next_states_prediction(next_states, finals, temp)
        targets = discount * next_states_prediction + rewards

        masked = copy.deepcopy(activations[-1])
        masked[np.arange(len(masked)), actions] = targets
        error = activations[-1] - masked
        return error

    def debug_magnitudes(self, bias_grad, weight_grad):
        self.weight0_gradient_tally += np.linalg.norm(weight_grad[0])
        norm = np.linalg.norm(weight_grad[1])
        self.weight1_gradient_tally += norm
        self.bias0_gradient_tally += np.linalg.norm(bias_grad[0])
        self.bias1_gradient_tally += np.linalg.norm(bias_grad[1])

        self.i += 1
        steps = 1
        if self.i == steps:
            add_value(self.weight0_gradient_tally / steps, "w0 magnitude")
            add_value(self.weight1_gradient_tally / steps, "w1 magnitude")
            add_value(self.bias0_gradient_tally / steps, "b0 magnitude")
            add_value(self.bias1_gradient_tally / steps, "b1 magnitude")

            self.i = 0
            self.weight0_gradient_tally = self.weight1_gradient_tally = self.bias0_gradient_tally = self.bias1_gradient_tally = 0

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
