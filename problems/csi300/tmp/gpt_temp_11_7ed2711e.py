import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Asymmetric Volatility-Weighted Turnover Momentum factor
    Combines price momentum, volatility asymmetry, and volume turnover signals
    """
    data = df.copy()
    
    # Parameters
    momentum_period = 10
    volatility_period = 20
    decay_factor = 0.9
    
    # Calculate exponential weights for momentum
    weights = np.array([decay_factor ** i for i in range(momentum_period)])[::-1]
    weights = weights / weights.sum()
    
    # Asymmetric Price Momentum Components
    # Upside momentum: High-Close based
    upside_momentum = np.zeros(len(data))
    for i in range(momentum_period, len(data)):
        period_data = data.iloc[i-momentum_period:i]
        high_close_ratios = (period_data['high'] / period_data['close'] - 1).values
        upside_momentum[i] = np.sum(weights * high_close_ratios)
    
    # Downside momentum: Low-Close based
    downside_momentum = np.zeros(len(data))
    for i in range(momentum_period, len(data)):
        period_data = data.iloc[i-momentum_period:i]
        low_close_ratios = (period_data['low'] / period_data['close'] - 1).values
        downside_momentum[i] = np.sum(weights * low_close_ratios)
    
    # Volatility Adjustment with Asymmetry
    # Upside volatility: High-Close range volatility
    upside_volatility = np.zeros(len(data))
    for i in range(volatility_period, len(data)):
        period_data = data.iloc[i-volatility_period:i]
        high_close_ranges = (period_data['high'] / period_data['close'] - 1).values
        upside_volatility[i] = np.std(high_close_ranges)
    
    # Downside volatility: Low-Close range volatility
    downside_volatility = np.zeros(len(data))
    for i in range(volatility_period, len(data)):
        period_data = data.iloc[i-volatility_period:i]
        low_close_ranges = (period_data['low'] / period_data['close'] - 1).values
        downside_volatility[i] = np.std(low_close_ranges)
    
    # Volume Intensity Analysis
    volume_change = np.zeros(len(data))
    for i in range(1, len(data)):
        if data['volume'].iloc[i-1] > 0:
            volume_change[i] = data['volume'].iloc[i] / data['volume'].iloc[i-1] - 1
    
    # Turnover-Momentum Interaction
    # Volume-weighted momentum adjustment
    volume_weighted_upside = upside_momentum * (1 + np.tanh(volume_change))
    volume_weighted_downside = downside_momentum * (1 + np.tanh(-volume_change))
    
    # Volatility-adjusted components
    volatility_adjusted_upside = np.zeros(len(data))
    volatility_adjusted_downside = np.zeros(len(data))
    
    for i in range(max(momentum_period, volatility_period), len(data)):
        if upside_volatility[i] > 0:
            volatility_adjusted_upside[i] = volume_weighted_upside[i] / upside_volatility[i]
        if downside_volatility[i] > 0:
            volatility_adjusted_downside[i] = volume_weighted_downside[i] / downside_volatility[i]
    
    # Price range normalization
    price_range = (data['high'] - data['low']) / data['close']
    open_close_range = abs(data['close'] - data['open']) / data['close']
    
    # Final Factor Construction
    factor = np.zeros(len(data))
    for i in range(max(momentum_period, volatility_period), len(data)):
        if abs(volatility_adjusted_downside[i]) > 0:
            # Ratio of enhanced upside vs downside signals with range context
            ratio_component = volatility_adjusted_upside[i] / abs(volatility_adjusted_downside[i])
            range_context = price_range.iloc[i] * open_close_range.iloc[i]
            factor[i] = ratio_component * (1 + range_context)
        else:
            factor[i] = volatility_adjusted_upside[i] * (1 + price_range.iloc[i])
    
    # Create output series
    factor_series = pd.Series(factor, index=data.index)
    
    return factor_series
