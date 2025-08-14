import savgol_filter
    smoothed_factor = pd.Series(savgol_filter(factor_values, window_length=11, polyorder=2), index=factor_values.index)
    heuristics_matrix = smoothed_factor.dropna()
    return heuristics_matrix
