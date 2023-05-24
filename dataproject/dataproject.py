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

def gini_coefficient(data):
    # Sort the data in ascending order
    sorted_data = np.sort(data)
    
    # Get the cumulative sum of the sorted data
    cumulative_sum = np.cumsum(sorted_data)
    
    # Calculate the Lorenz curve
    Lorenz_curve = cumulative_sum / np.sum(sorted_data)
    
    # Calculate the area under the Lorenz curve using the trapezoidal rule
    area_lorenz = np.trapz(Lorenz_curve)
    
    # Calculate the area of the perfect equality line
    area_perfect_equality = np.linspace(0, 1, len(data) + 1)[1:]
    
    # Calculate the Gini coefficient
    gini = 1 - 2 * area_lorenz
    
    # Make sure the Gini coefficient is within the range of 0 to 1
    gini = max(0, min(1, gini))
    
    # Return the Gini coefficient
    return gini



# Define a function to calculate the Gini coefficient
def gini(x):
    total = 0
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (len(x)**2 * np.mean(x))