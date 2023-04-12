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
