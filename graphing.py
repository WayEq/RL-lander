import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

plt.style.use('ggplot')

avgs = []

graph_range = 300
index = range(300)
avgs = np.random.randn(len(index))
rep_freq = 10
line1 = []

def graph_average_reward(avg_reward):
    global avgs, epsilon, line1
    print("Avg reward now ", avg_reward)
    avgs = np.append(avgs[1:], float(avg_reward))
    line1 = live_plotter(index, avgs, line1)
    # epsilon = 1 - ((avg_reward + 20200) / 20400)
    # avg_reward = 0

def live_plotter(x_vec, y1_data, line1, identifier='', pause_time=0.1):
    if line1 == []:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig = plt.figure(figsize=(13, 6))
        ax = fig.add_subplot(111)
        # create a variable for the line so we can later update it
        line1, = ax.plot(x_vec, y1_data, '-o', alpha=0.8)
        # update plot label/title
        plt.ylabel('Y Label')
        plt.title('Title: {}'.format(identifier))
        plt.show()

    # after the figure, axis, and line are created, we only need to update the y-data
    line1.set_ydata(y1_data)
    # adjust limits if new data goes beyond bounds
    if np.min(y1_data) <= line1.axes.get_ylim()[0] or np.max(y1_data) >= line1.axes.get_ylim()[1]:
        plt.ylim([np.min(y1_data) - np.std(y1_data), np.max(y1_data) + np.std(y1_data)])
    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
    plt.pause(pause_time)

    # return line so we can update it again in the next iteration
    return line1