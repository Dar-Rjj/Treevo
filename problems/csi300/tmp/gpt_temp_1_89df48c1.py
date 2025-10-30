import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Multi-Timeframe Regime Detection
    # Short-Term Regime (5-day)
    data['price_momentum'] = data['close'] / data['close'].shift(5) - 1
    data['volume_momentum'] = data['volume'] / data['volume'].shift(5)
    
    # Intraday Regime
    data['opening_pressure'] = (data['open'] - data['close'].shift(1)) / (data['high'] - data['low'])
    data['intraday_efficiency'] = abs(data['close'] - data['open']) / (data['high'] - data['low'])
    
    # Volume-Price Divergence System
    # Efficiency Divergence
    data['volume_efficiency'] = data['volume'] / abs(data['close'] - data['close'].shift(1))
    data['volume_efficiency'] = data['volume_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    # Efficiency Trend (3-day slope using linear regression)
    def calc_slope(series):
        if len(series) < 3 or series.isna().any():
            return np.nan
        x = np.arange(len(series))
        return np.polyfit(x, series.values, 1)[0]
    
    data['efficiency_trend'] = data['volume_efficiency'].rolling(window=3, min_periods=3).apply(calc_slope, raw=False)
    
    # Concentration Divergence
    # Since we don't have volume at specific price levels, we'll use proxies
    data['high_low_volume_ratio'] = (data['volume'] * (data['close'] - data['low'])) / (data['volume'] * (data['high'] - data['close']))
    data['high_low_volume_ratio'] = data['high_low_volume_ratio'].replace([np.inf, -np.inf], np.nan)
    data['volume_to_range_ratio'] = data['volume'] / (data['high'] - data['low'])
    
    # Amount-Based Microstructure Signals
    data['implied_price_deviation'] = abs((data['amount']/data['volume']) - (data['high']+data['low'])/2) / (data['high']-data['low'])
    data['volatility_efficiency'] = abs(data['close'] - data['close'].shift(1)) / (data['high'] - data['low'])
    
    # Alpha Signal Generation
    # Regime Alignment
    data['regime_alignment'] = np.sign(data['price_momentum']) * np.sign(data['intraday_efficiency'])
    
    # Volume Confirmation (positive when volume supports price movement)
    data['volume_confirmation'] = np.sign(data['price_momentum']) * data['volume_momentum']
    
    # Divergence Strength (combining multiple divergence measures)
    data['divergence_strength'] = (
        -data['efficiency_trend'] +  # Negative trend indicates divergence
        (1 - data['high_low_volume_ratio'].clip(0.1, 10)) +  # Extreme ratios indicate divergence
        (1 - data['volume_to_range_ratio'] / data['volume_to_range_ratio'].rolling(10).mean())
    )
    
    # Signal Generation Logic
    conditions = [
        # Strong Continuation Signal
        (data['regime_alignment'] > 0) & (data['volume_confirmation'] > 0),
        
        # Reversal Signal
        (data['divergence_strength'] < -0.1) & (data['implied_price_deviation'] > 0.5),
        
        # Weak/No Signal (default)
        True
    ]
    
    choices = [1, -1, 0]  # 1 for continuation, -1 for reversal, 0 for no signal
    
    data['alpha_signal'] = np.select(conditions, choices, default=0)
    
    # Final factor: weighted combination of signals and underlying metrics
    data['factor'] = (
        data['alpha_signal'] * 0.4 +
        data['regime_alignment'] * 0.2 +
        data['divergence_strength'] * 0.2 +
        data['volume_confirmation'] * 0.1 +
        data['implied_price_deviation'] * 0.1
    )
    
    return data['factor']
