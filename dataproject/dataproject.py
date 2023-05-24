import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets


# Define a function to calculate the Lorenz curve
def lorenz_curve(data):
    # Get the sorted data
    sorted_data = np.sort(data)
    # Calculate the cumulative sums of the data
    cumsum_data = np.cumsum(sorted_data)
    # Calculate the cumulative percentage of the data
    cumperc_data = 100 * cumsum_data / cumsum_data[-1]
    # Calculate the equally distributed population percentages
    pop_perc = np.linspace(0, 100, len(data))
    # Return the cumulative percentage of the data and the equally distributed population percentages
    return pop_perc, cumperc_data

# Define a function to calculate the Gini coefficient
def gini(x):
    total = 0
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (len(x)**2 * np.mean(x))