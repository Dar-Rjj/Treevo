import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate Multi-Timeframe Price Fractal
    # Micro-Fractal Pattern (3-day)
    micro_fractal = (data['high'] - data['low']) / (data['high'].shift(2) - data['low'].shift(2))
    
    # Macro-Fractal Pattern (10-day)
    macro_fractal = (data['high'] - data['low']) / (data['high'].shift(9) - data['low'].shift(9))
    
    # Fractal Momentum Divergence
    fractal_momentum = micro_fractal / macro_fractal
    
    # Analyze Liquidity Regime Switching
    # Volume Regime Indicator
    volume_window = data['volume'].rolling(window=5, min_periods=1)
    volume_percentile = data['volume'].rolling(window=6, min_periods=1).apply(
        lambda x: (x.iloc[-1] - x.iloc[:-1].min()) / (x.iloc[:-1].max() - x.iloc[:-1].min()) if len(x) > 1 and (x.iloc[:-1].max() - x.iloc[:-1].min()) > 0 else 0.5
    )
    
    # Amount Efficiency Regime
    amount_efficiency = data['amount'] / data['volume']
    amount_efficiency_window = amount_efficiency.rolling(window=5, min_periods=1)
    amount_efficiency_percentile = amount_efficiency.rolling(window=6, min_periods=1).apply(
        lambda x: (x.iloc[-1] - x.iloc[:-1].min()) / (x.iloc[:-1].max() - x.iloc[:-1].min()) if len(x) > 1 and (x.iloc[:-1].max() - x.iloc[:-1].min()) > 0 else 0.5
    )
    
    # Liquidity Regime Composite
    liquidity_regime = volume_percentile * amount_efficiency_percentile
    
    # Calculate Price Momentum Persistence
    returns = data['close'].pct_change()
    momentum_persistence = returns.rolling(window=5, min_periods=1).apply(
        lambda x: _calculate_directional_streak(x), raw=False
    )
    
    # Construct Regime-Adaptive Factor
    factor = fractal_momentum * liquidity_regime * momentum_persistence
    
    return factor

def _calculate_directional_streak(series):
    if len(series) < 2:
        return 1
    
    # Get the last 5 elements (current and previous 4)
    recent_returns = series.iloc[-5:] if len(series) >= 5 else series
    
    # Calculate longest streak of same-direction returns
    current_streak = 1
    max_streak = 1
    
    for i in range(1, len(recent_returns)):
        if (recent_returns.iloc[i] > 0 and recent_returns.iloc[i-1] > 0) or \
           (recent_returns.iloc[i] < 0 and recent_returns.iloc[i-1] < 0) or \
           (recent_returns.iloc[i] == 0 and recent_returns.iloc[i-1] == 0):
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 1
    
    return max_streak
