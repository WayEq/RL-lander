import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

plt.style.use('ggplot')

values = {}

# np.random.seed(1)
graph_range = 300

reinit = False
def add_value(value, title):
    global values, reinit

    if not title in values:
        values[title] = []
        reinit = True

    values[title] = np.append(values[title], float(value))

    if len(values[title]) > graph_range:
        values[title] = values[title][1:]

ax_global = None
fig = None
def live_plotter(pause_time=0.01):
    global lines, fig, ax_global, reinit
    num_plots = len(values)
    if fig is None or reinit:
        if fig is not None:
            plt.close(fig)
        fig, ax_global = plt.subplots(num_plots, figsize=(10,5))
        reinit = False

    i = 0

    for key, value in values.items():
        ax = ax_global
        if num_plots > 1:
            ax = ax_global[i]
        ax.clear()
        ax.plot(values[key])
        ax.set_title(key)
        i += 1
    plt.ion()

    plt.show()
    plt.pause(pause_time)
