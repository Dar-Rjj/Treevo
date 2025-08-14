def weighted_moving_average(data, weights):
    return (data * weights).sum() / weights.sum()
