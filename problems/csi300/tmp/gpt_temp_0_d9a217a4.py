import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Momentum Decay
    # Short-term momentum decay (1-3 days)
    short_term_momentum = (data['close'] - data['close'].shift(1)) - (data['close'].shift(1) - data['close'].shift(2))
    
    # Medium-term momentum decay (5-10 days)
    medium_term_momentum = ((data['close'] / data['close'].shift(5) - 1) - 
                           (data['close'].shift(5) / data['close'].shift(10) - 1))
    
    # Long-term momentum decay (20+ days)
    roc_20 = (data['close'] / data['close'].shift(20) - 1)
    roc_40 = (data['close'] / data['close'].shift(40) - 1)
    long_term_momentum = roc_20 - roc_40
    
    # Volume-Price Divergence
    # Intraday divergence
    intraday_price_move = data['close'] - data['open']
    intraday_range = data['high'] - data['low']
    intraday_divergence = intraday_price_move / (intraday_range + 1e-8)
    
    # Multi-day divergence (t-4 to t)
    multi_day_divergence = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 4:
            window_data = data.iloc[i-4:i+1]
            price_div = (window_data['close'] - window_data['open']).sum()
            range_div = (window_data['high'] - window_data['low']).sum()
            multi_day_divergence.iloc[i] = price_div / (range_div + 1e-8)
    
    # Cross-timeframe divergence (3-day vs 8-day)
    def calculate_divergence(window_size):
        divergence = pd.Series(index=data.index, dtype=float)
        for i in range(len(data)):
            if i >= window_size - 1:
                window_data = data.iloc[i-window_size+1:i+1]
                price_sum = (window_data['close'] - window_data['open']).sum()
                range_sum = (window_data['high'] - window_data['low']).sum()
                divergence.iloc[i] = price_sum / (range_sum + 1e-8)
        return divergence
    
    divergence_3d = calculate_divergence(3)
    divergence_8d = calculate_divergence(8)
    cross_timeframe_divergence = divergence_3d - divergence_8d
    
    # Price Efficiency Decay
    # Daily efficiency
    daily_efficiency = abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # Momentum of efficiency
    efficiency_momentum = daily_efficiency - daily_efficiency.shift(1)
    
    # Multi-period efficiency (3-day vs 10-day)
    efficiency_3d = daily_efficiency.rolling(window=3, min_periods=3).mean()
    efficiency_10d = daily_efficiency.rolling(window=10, min_periods=10).mean()
    multi_period_efficiency = efficiency_3d - efficiency_10d
    
    # Volume Distribution
    # Volume momentum vs price momentum correlation (3-day rolling)
    volume_change = data['volume'].pct_change()
    price_change = data['close'].pct_change()
    volume_price_corr = volume_change.rolling(window=3, min_periods=3).corr(price_change)
    
    # Multi-timeframe volume trends (3-day vs 8-day)
    volume_3d_trend = data['volume'].rolling(window=3, min_periods=3).mean()
    volume_8d_trend = data['volume'].rolling(window=8, min_periods=8).mean()
    volume_trend_divergence = volume_3d_trend / (volume_8d_trend + 1e-8) - 1
    
    # Composite Factor
    # Normalize components
    components = {
        'short_momentum': short_term_momentum,
        'medium_momentum': medium_term_momentum,
        'long_momentum': long_term_momentum,
        'intraday_div': intraday_divergence,
        'multi_day_div': multi_day_divergence,
        'cross_div': cross_timeframe_divergence,
        'efficiency': daily_efficiency,
        'eff_momentum': efficiency_momentum,
        'multi_eff': multi_period_efficiency,
        'vol_price_corr': volume_price_corr,
        'vol_trend_div': volume_trend_divergence
    }
    
    # Normalize each component
    normalized_components = {}
    for name, series in components.items():
        if series.notna().any():
            mean_val = series.mean()
            std_val = series.std()
            if std_val > 0:
                normalized_components[name] = (series - mean_val) / std_val
            else:
                normalized_components[name] = series - mean_val
    
    # Calculate composite factor with weights
    momentum_weight = 0.3
    divergence_weight = 0.25
    efficiency_weight = 0.25
    volume_weight = 0.2
    
    composite_factor = (
        momentum_weight * (
            normalized_components.get('short_momentum', 0) + 
            normalized_components.get('medium_momentum', 0) + 
            normalized_components.get('long_momentum', 0)
        ) +
        divergence_weight * (
            normalized_components.get('intraday_div', 0) + 
            normalized_components.get('multi_day_div', 0) + 
            normalized_components.get('cross_div', 0)
        ) +
        efficiency_weight * (
            normalized_components.get('efficiency', 0) + 
            normalized_components.get('eff_momentum', 0) + 
            normalized_components.get('multi_eff', 0)
        ) +
        volume_weight * (
            normalized_components.get('vol_price_corr', 0) + 
            normalized_components.get('vol_trend_div', 0)
        )
    )
    
    return composite_factor
