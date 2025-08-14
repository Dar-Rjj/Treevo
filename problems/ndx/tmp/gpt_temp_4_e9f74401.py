import seasonal_decompose
    result = seasonal_decompose(combined_returns.dropna(), model='additive', period=5)
    seasonal_component = result.seasonal
    
    heuristics_matrix = seasonal_component
    return heuristics_matrix
