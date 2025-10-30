import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum Component
    # Short-Term Momentum (5-day)
    data['short_return'] = data['close'] / data['close'].shift(5) - 1
    data['short_vol_weighted_return'] = (data['close'] / data['close'].shift(5) - 1) * (data['volume'] / data['volume'].shift(5))
    
    # Medium-Term Momentum (10-day)
    data['medium_return'] = data['close'] / data['close'].shift(10) - 1
    data['momentum_divergence'] = data['short_return'] - data['medium_return']
    
    # Momentum Interaction
    data['momentum_interaction'] = data['short_return'] * data['medium_return']
    
    # Volume-Price Dynamics
    # Volume Acceleration
    def volume_slope(series):
        if len(series) < 3:
            return np.nan
        x = np.arange(len(series))
        slope, _, _, _, _ = stats.linregress(x, series)
        return slope
    
    data['volume_slope'] = data['volume'].rolling(window=3, min_periods=3).apply(volume_slope, raw=True)
    data['volume_ratio'] = data['volume'] / data['volume'].shift(5)
    
    # Price-Volume Divergence
    data['price_volume_divergence'] = data['short_return'] * (1 / data['volume_ratio'])
    
    # Volume Confirmation Strength
    data['volume_confirmation'] = data['volume_slope'] * data['price_volume_divergence']
    
    # Volatility Context
    # Trading Range Analysis
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['avg_true_range'] = data['true_range'].rolling(window=10, min_periods=10).mean()
    
    # High-low range persistence
    data['high_low_range'] = (data['high'] - data['low']) / data['close']
    data['range_persistence'] = data['high_low_range'].rolling(window=5, min_periods=5).std()
    
    # Volatility Adjustment
    data['volatility_adjusted_momentum'] = data['momentum_interaction'] / data['avg_true_range']
    
    # Intraday Dynamics
    # Gap Analysis
    data['opening_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['intraday_recovery'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Intraday Efficiency
    data['price_impact'] = (data['high'] - data['low']) / data['amount'].replace(0, np.nan)
    data['intraday_efficiency'] = data['opening_gap'] * data['intraday_recovery'] * (1 / data['price_impact'])
    
    # Signal Integration
    # Primary Signal Generation
    data['primary_signal'] = data['volatility_adjusted_momentum'] * data['volume_confirmation'] * data['intraday_efficiency']
    
    # Contrarian Logic
    signal_magnitude = data['primary_signal'].abs()
    signal_persistence = data['primary_signal'].rolling(window=3, min_periods=3).std()
    
    # Apply reversal expectation for extreme signals
    extreme_threshold = data['primary_signal'].rolling(window=20, min_periods=20).quantile(0.9)
    data['contrarian_adjustment'] = np.where(
        data['primary_signal'].abs() > extreme_threshold,
        -1 * data['primary_signal'] * (signal_magnitude / signal_persistence).replace(np.inf, 1),
        1
    )
    
    # Final Alpha Factor
    data['alpha_factor'] = (
        data['momentum_interaction'] * 
        data['volume_confirmation'] * 
        data['intraday_efficiency'] / 
        data['avg_true_range']
    ) * data['contrarian_adjustment']
    
    return data['alpha_factor']
