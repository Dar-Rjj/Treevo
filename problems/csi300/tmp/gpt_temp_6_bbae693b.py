import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(df):
    """
    Multi-Timeframe Momentum-Volume Divergence with Acceleration Analysis
    Generates alpha factor combining momentum divergence, volume acceleration,
    volatility adjustment, and contrarian logic
    """
    data = df.copy()
    
    # Multi-Timeframe Momentum Divergence
    # Short-Term Momentum (5-day)
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_5d_strength'] = data['momentum_5d'].abs()
    
    # Medium-Term Momentum (10-day)
    data['momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    data['momentum_persistence'] = data['momentum_10d'].rolling(window=5).std()
    
    # Momentum Divergence Signal
    data['momentum_divergence'] = data['momentum_5d'] * data['momentum_10d']
    
    # Volume Acceleration & Confirmation
    # Volume Trend Analysis using 3-day linear regression
    def calc_volume_slope(volume_series):
        if len(volume_series) < 3:
            return np.nan
        X = np.arange(len(volume_series)).reshape(-1, 1)
        y = volume_series.values
        model = LinearRegression()
        model.fit(X, y)
        return model.coef_[0]
    
    data['volume_slope'] = data['volume'].rolling(window=3).apply(
        calc_volume_slope, raw=False
    )
    
    # Relative Volume Strength
    data['avg_volume_10d'] = data['volume'].rolling(window=10).mean()
    data['relative_volume'] = data['volume'] / data['avg_volume_10d']
    
    # Volume-Momentum Divergence
    data['volume_momentum_div'] = data['momentum_divergence'] / (data['volume_slope'] + 1e-8)
    
    # Volatility-Adjusted Signal Integration
    # Average True Range (ATR) calculation
    data['tr'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['atr_10d'] = data['tr'].rolling(window=10).mean()
    
    # Signal Volatility Scaling
    data['volatility_scaled_signal'] = data['volume_momentum_div'] / (data['atr_10d'] + 1e-8)
    
    # Volume-Weighted Enhancement
    data['volume_weighted_signal'] = data['volatility_scaled_signal'] * data['relative_volume']
    
    # Contrarian Logic Application
    # Extreme Signal Detection
    signal_series = data['volume_weighted_signal'].dropna()
    if len(signal_series) > 0:
        signal_mean = signal_series.mean()
        signal_std = signal_series.std()
        data['signal_zscore'] = (data['volume_weighted_signal'] - signal_mean) / (signal_std + 1e-8)
        
        # Reversal Probability Assessment
        data['contrarian_signal'] = data['volume_weighted_signal'].copy()
        
        # Apply negative scaling to extreme positive signals
        extreme_positive_mask = data['signal_zscore'] > 2
        data.loc[extreme_positive_mask, 'contrarian_signal'] = (
            data.loc[extreme_positive_mask, 'volume_weighted_signal'] * -0.5
        )
        
        # Apply positive scaling to extreme negative signals
        extreme_negative_mask = data['signal_zscore'] < -2
        data.loc[extreme_negative_mask, 'contrarian_signal'] = (
            data.loc[extreme_negative_mask, 'volume_weighted_signal'] * -1.5
        )
    else:
        data['contrarian_signal'] = data['volume_weighted_signal']
    
    # Final Alpha Factor Generation
    alpha_factor = data['contrarian_signal']
    
    return alpha_factor
