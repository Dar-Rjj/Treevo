import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Compute Price Momentum Components
    # Short-term Momentum
    data['short_momentum'] = data['close'] / data['close'].shift(5) - 1
    data['daily_range'] = data['high'] - data['low']
    data['range_momentum'] = data['daily_range'] / data['daily_range'].shift(1) - 1
    
    # Medium-term Momentum
    data['medium_momentum'] = data['close'] / data['close'].shift(10) - 1
    data['abs_close_momentum'] = abs(data['close'] / data['close'].shift(1) - 1)
    
    # Identify Momentum Divergence Pattern
    data['momentum_diff'] = data['short_momentum'] - data['medium_momentum']
    data['range_expansion'] = data['range_momentum'] * data['abs_close_momentum']
    data['divergence_strength'] = data['momentum_diff'] * data['range_expansion']
    
    # Volume Confirmation Analysis
    data['volume_ratio'] = data['volume'] / data['volume'].shift(1)
    data['volume_acceleration'] = data['volume_ratio'] / data['volume_ratio'].shift(1) - 1
    data['volume_support'] = data['volume_ratio'] * data['volume_acceleration']
    data['price_volume_component'] = data['divergence_strength'] * data['volume_support']
    
    # Volatility Regime Classification
    data['returns'] = data['close'].pct_change()
    data['rolling_vol_20d'] = data['returns'].rolling(window=20, min_periods=10).std()
    
    # Calculate volatility percentiles using 60-day rolling window
    vol_percentiles = data['rolling_vol_20d'].rolling(window=60, min_periods=30).apply(
        lambda x: pd.Series(x).quantile(0.8) if len(x) >= 30 else np.nan, raw=False
    )
    vol_percentiles_low = data['rolling_vol_20d'].rolling(window=60, min_periods=30).apply(
        lambda x: pd.Series(x).quantile(0.2) if len(x) >= 30 else np.nan, raw=False
    )
    
    # Classify volatility regimes
    conditions = [
        data['rolling_vol_20d'] > vol_percentiles,
        data['rolling_vol_20d'] < vol_percentiles_low
    ]
    choices = ['high', 'low']
    data['vol_regime'] = np.select(conditions, choices, default='normal')
    
    # Volatility Adjustment
    def volatility_adjustment(row):
        if row['vol_regime'] == 'high':
            return np.sqrt(row['rolling_vol_20d']) if row['rolling_vol_20d'] > 0 else 0
        elif row['vol_regime'] == 'low':
            return row['rolling_vol_20d']
        else:  # normal
            return np.log1p(row['rolling_vol_20d']) if row['rolling_vol_20d'] > 0 else 0
    
    data['vol_adjustment'] = data.apply(volatility_adjustment, axis=1)
    
    # Generate Adaptive Factor
    # Preserve original sign from price divergence
    sign_preservation = np.sign(data['divergence_strength'])
    data['factor'] = data['price_volume_component'] * data['vol_adjustment'] * sign_preservation
    
    # Return the factor series
    return data['factor']
