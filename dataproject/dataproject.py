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
def gini_coefficient(data):
    # Calculate the Lorenz curve
    x, y = lorenz_curve(data)
    # Calculate the area between the Lorenz curve and the diagonal line of perfect equality
    area = np.trapz(y, x) - 0.5
    # Calculate the total area under the diagonal line of perfect equality
    total_area = 0.5 * x[-1]
    # Calculate the Gini coefficient as half of the ratio of the area between the Lorenz curve and the diagonal line to the total area
    gini = area / total_area
    # Return the Gini coefficient
    return gini
