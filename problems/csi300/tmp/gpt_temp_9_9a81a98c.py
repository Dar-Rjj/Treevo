import numpy as np
def heuristics_v2(df):
    # Calculate 10-day and 5-day exponential weights
    def exp_weights(n):
        return np.exp(np.linspace(0, -1, n)) / np.sum(np.exp(np.linspace(0, -1, n)))
