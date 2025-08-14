import pandas as pd
    # Calculate recent price momentum
    momentum = df['close'].pct_change(periods=5)
    # Compute volume divergence from its 20-day moving average
    volume_ma = df['volume'].rolling(window=20).mean()
    volume_divergence = (df['volume'] - volume_ma) / volume_ma
    # Combine factors into a simple heuristic: 70% momentum, 30% volume divergence
    heuristics_matrix = 0.7 * momentum + 0.3 * volume_divergence
    return heuristics_matrix
