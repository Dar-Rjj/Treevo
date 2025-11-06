import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Momentum-Volume Fractal Breakout Alpha Factor
    Combines multi-timeframe momentum, volume-price efficiency, congestion breakout detection,
    and order flow imbalance to generate trading signals.
    """
    data = df.copy()
    
    # Multi-timeframe Momentum Analysis
    data['short_roc'] = data['close'] / data['close'].shift(5) - 1
    data['medium_roc'] = data['close'] / data['close'].shift(10) - 1
    data['long_roc'] = data['close'] / data['close'].shift(20) - 1
    
    # Momentum Pattern Complexity
    data['short_med_divergence'] = data['short_roc'] - data['medium_roc']
    data['med_long_divergence'] = data['medium_roc'] - data['long_roc']
    data['momentum_acceleration'] = (data['short_roc'] - data['medium_roc']) - (data['medium_roc'] - data['long_roc'])
    
    # Volume-Price Fractal Efficiency
    data['daily_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['efficiency_5d_avg'] = data['daily_efficiency'].rolling(window=5).mean()
    data['efficiency_momentum'] = data['daily_efficiency'] - data['efficiency_5d_avg']
    
    # Volume Distribution Pattern
    data['volume_5d_avg'] = data['volume'].rolling(window=5).mean()
    data['volume_spike_ratio'] = data['volume'] / data['volume_5d_avg']
    
    # Volume-Price Correlation Fractal
    data['price_change'] = data['close'] - data['close'].shift(1)
    data['abs_price_change'] = data['price_change'].abs()
    data['volume_price_corr'] = data['volume'].rolling(window=10).corr(data['abs_price_change'])
    
    # Congestion Breakout Detection
    data['daily_range'] = data['high'] - data['low']
    data['range_5d_avg'] = data['daily_range'].rolling(window=5).mean()
    data['range_compression'] = data['daily_range'] / data['range_5d_avg']
    
    # Breakout Signal Generation
    data['prev_5d_high'] = data['high'].rolling(window=5).apply(lambda x: x[:-1].max() if len(x) > 1 else np.nan)
    data['prev_5d_low'] = data['low'].rolling(window=5).apply(lambda x: x[:-1].min() if len(x) > 1 else np.nan)
    
    data['upper_breakout'] = (data['close'] > data['prev_5d_high']).astype(float)
    data['lower_breakout'] = (data['close'] < data['prev_5d_low']).astype(float)
    
    data['upper_breakout_strength'] = np.where(
        data['upper_breakout'] == 1, 
        (data['close'] - data['prev_5d_high']) / data['daily_range'], 
        0
    )
    data['lower_breakout_strength'] = np.where(
        data['lower_breakout'] == 1, 
        (data['prev_5d_low'] - data['close']) / data['daily_range'], 
        0
    )
    
    # Order Flow Imbalance Integration
    data['up_move_volume'] = np.where(data['close'] > data['close'].shift(1), data['volume'], 0)
    data['down_move_volume'] = np.where(data['close'] < data['close'].shift(1), data['volume'], 0)
    
    data['up_volume_5d'] = data['up_move_volume'].rolling(window=5).sum()
    data['down_volume_5d'] = data['down_move_volume'].rolling(window=5).sum()
    data['pressure_ratio'] = data['up_volume_5d'] / (data['up_volume_5d'] + data['down_volume_5d']).replace(0, np.nan)
    
    data['pressure_5d_avg'] = data['pressure_ratio'].rolling(window=5).mean()
    data['pressure_trend'] = data['pressure_ratio'] - data['pressure_5d_avg']
    
    # Alpha Signal Synthesis
    # Strong Trend Continuation Component
    trend_strength = (
        data['short_roc'].rank(pct=True) * 0.3 +
        data['volume_spike_ratio'].rank(pct=True) * 0.3 +
        data['daily_efficiency'].rank(pct=True) * 0.2 +
        data['pressure_trend'].rank(pct=True) * 0.2
    )
    
    # Breakout Confirmation Component
    breakout_signal = (
        (data['upper_breakout_strength'] - data['lower_breakout_strength']).rank(pct=True) * 0.4 +
        data['volume_spike_ratio'].rank(pct=True) * 0.3 +
        data['pressure_trend'].rank(pct=True) * 0.3
    )
    
    # Reversal Pattern Component
    reversal_signal = (
        (data['momentum_acceleration'] * -1).rank(pct=True) * 0.4 +
        (data['volume_price_corr'] * -1).rank(pct=True) * 0.3 +
        (data['pressure_trend'] * -1).rank(pct=True) * 0.3
    )
    
    # Consolidation Signal Component
    consolidation_signal = (
        (data['range_compression'] * -1).rank(pct=True) * 0.5 +
        (data['volume_spike_ratio'] * -1).rank(pct=True) * 0.5
    )
    
    # Final Alpha Factor
    alpha_factor = (
        trend_strength * 0.35 +
        breakout_signal * 0.30 +
        reversal_signal * 0.20 +
        consolidation_signal * 0.15
    )
    
    return alpha_factor
