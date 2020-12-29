import numpy as np
#
from nn import NN
from scipy.special import softmax
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_printoptions(precision=10)

class TorchNet(nn.Module):

    def __init__(self, weights, bias):
        super(TorchNet, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 2)

        with torch.no_grad():
            self.fc1.weight = nn.Parameter(torch.from_numpy(weights[0].T))
            self.fc1.bias = nn.Parameter(torch.from_numpy(bias[0]))
            self.fc2.weight = nn.Parameter(torch.from_numpy(weights[1].T))
            self.fc2.bias = nn.Parameter(torch.from_numpy(bias[1]))


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x




def test_softmax():

    temp = .001
    x = np.array([[1, 2, 3, 4]])
    output = NN.softmax(x, temp)
    npvalue = softmax(x/ temp, axis=1)
    if not np.allclose(output, npvalue):
        raise ValueError("softmax")


def run_tests():
    test_feed_forward()
    test_calc_output_grad()
    test_softmax()
    test_update_deltas()


def test_feed_forward():
    config = get_config()
    num_inputs = config["layer node count"][0]
    num_hidden = config["layer node count"][1]
    num_output = config["layer node count"][2]
    for i in range(5):
        weights = [np.random.randn(num_inputs, num_hidden), np.random.randn(num_hidden, num_output)]
        config["weights"] = weights
        bias = [np.random.randn(num_hidden), np.random.randn(num_output)]
        config["bias"] = bias


        state = np.random.randn(num_inputs)
        state2 = np.random.randn(num_inputs)
        state3 = np.random.randn(num_inputs)
        states = np.row_stack((state, state2, state3))

        nn = NN(config)
        net = TorchNet(weights, bias)


        zs, activations = nn.feed_forward(states)



        out = net(torch.from_numpy(states))
        if not torch.allclose(out, torch.from_numpy(activations[-1])):
            print (out, "\n", torch.from_numpy(activations[-1]))
            raise ValueError("feed forward doesn't match torch")


def test_update_deltas():
    config = get_config()

    num_inputs = config["layer node count"][0]
    num_hidden = config["layer node count"][1]
    num_output = config["layer node count"][2]
    weights = [np.random.rand(num_inputs, num_hidden), np.random.rand(num_hidden, num_output)]
    config["weights"] = weights
    bias = [np.random.rand(num_hidden), np.random.rand(num_output)]
    config["bias"] = bias

    states = get_random_states(num_inputs)
    next_states = get_random_states(num_inputs)

    nn = NN(config)


    net = TorchNet(weights, bias)
    out = net(torch.from_numpy(states))
    out.backward(torch.tensor([[1,2],[0,1],[1,3]]))

    parameters = net.parameters()
    batch_bias_grad = []
    batch_weight_grad = []
    batch_weight_grad.append(net.fc1.weight.grad.numpy())
    batch_weight_grad.append(net.fc2.weight.grad.numpy())
    batch_bias_grad.append(net.fc1.bias.grad.numpy())
    batch_bias_grad.append(net.fc2.bias.grad.numpy())
    update_w_deltas, update_b_deltas = nn.get_update_deltas(batch_weight_grad, batch_bias_grad)

    optimizer = torch.optim.Adam(parameters, lr=0.01)
    old_bias1 = torch.clone(net.fc1.bias)
    old_bias2 = torch.clone(net.fc2.bias)
    old_weight1 = torch.clone(net.fc1.weight)
    old_weight2 = torch.clone(net.fc2.weight)
    optimizer.step()
    pt_delta = old_bias1 - net.fc1.bias
    pt_delta2 = old_bias2 - net.fc2.bias
    if not torch.allclose(pt_delta, torch.from_numpy(update_b_deltas[0])) or not torch.allclose(pt_delta2, torch.from_numpy(update_b_deltas[1])):
        raise ValueError("bias update off")
    pt_delta = old_weight1 - net.fc1.weight
    pt_delta2 = old_weight2 - net.fc2.weight
    if not torch.allclose(pt_delta, torch.from_numpy(update_w_deltas[0])) or not torch.allclose(pt_delta2, torch.from_numpy(update_w_deltas[1])):
        raise ValueError("weight update off")


def test_calc_output_grad():
    config = get_config()

    num_inputs = config["layer node count"][0]
    num_hidden = config["layer node count"][1]
    num_output = config["layer node count"][2]
    for i in range(5):
        weights = [np.random.rand(num_inputs, num_hidden), np.random.rand(num_hidden, num_output)]
        config["weights"] = weights
        bias = [np.random.rand(num_hidden), np.random.rand(num_output)]
        config["bias"] = bias

        states = get_random_states(num_inputs)

        nn = NN(config)

        zs, activations = nn.feed_forward(states)

        errors = [[1, 0],[1, 0], [1, 0]]
        bg, wg = nn.calc_gradients(states, np.array(errors), zs, activations)

        net = TorchNet(weights, bias)
        out = net(torch.from_numpy(states))
        with torch.no_grad():
            targets = out - torch.tensor(errors)

        out = ((targets - out) ** 2) / 2
        out = torch.sum(out, dim=1)
        out = torch.mean(out)
        out.backward()
        pt_w0_grad = net.fc1.weight.grad
        pt_w1_grad = net.fc2.weight.grad
        pt_b0_grad = net.fc1.bias.grad
        pt_b1_grad = net.fc2.bias.grad
        if not torch.allclose(pt_w0_grad, torch.from_numpy(wg[0].T)) \
                or not torch.allclose(pt_w1_grad, torch.from_numpy(wg[1].T)) \
                or not torch.allclose(pt_b0_grad, torch.from_numpy(bg[0])) \
                or not torch.allclose(pt_b1_grad, torch.from_numpy(bg[1])):
            raise ValueError("shit")


def get_random_states(num_inputs):
    state = np.random.rand(num_inputs)
    state2 = np.random.rand(num_inputs)
    state3 = np.random.rand(num_inputs)
    states = np.row_stack((state, state2, state3))
    return states


def get_config():
    config = {
        "learning rate": .01,
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
    return config


# run_tests()