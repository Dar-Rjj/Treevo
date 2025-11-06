import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Adjusted Range Efficiency with Volume Confirmation factor
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate True Range
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Calculate Range Efficiency Component
    # Net price movement over 5 days
    data['net_movement'] = data['close'] - data['close'].shift(5)
    
    # Total oscillatory movement (sum of true ranges over 5 days)
    data['total_oscillatory'] = data['true_range'].rolling(window=5, min_periods=3).sum()
    
    # Price movement efficiency
    data['range_efficiency'] = data['net_movement'] / data['total_oscillatory']
    data['range_efficiency'] = data['range_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    # Volume Confirmation Layer
    # Volume momentum divergence
    data['price_return_5d'] = data['close'].pct_change(5)
    data['volume_change_5d'] = data['volume'].pct_change(5)
    
    # Avoid division by zero and handle infinite values
    data['volume_change_5d_adj'] = data['volume_change_5d'].replace([np.inf, -np.inf], np.nan)
    data['inv_volume_change'] = 1 / (1 + abs(data['volume_change_5d_adj']))
    data['volume_divergence'] = data['price_return_5d'] * data['inv_volume_change']
    
    # Volume-regime weighting
    data['volume_ma_20'] = data['volume'].rolling(window=20, min_periods=10).mean()
    data['volume_ratio'] = data['volume'] / data['volume_ma_20']
    
    # Volume percentile rank over 20-day window
    data['volume_percentile'] = data['volume'].rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
    )
    
    # Dynamic volume weighting
    data['volume_weight'] = np.where(
        data['volume_ratio'] > 1.2,
        data['volume_percentile'] * 1.5,  # Enhance during high volume
        np.where(
            data['volume_ratio'] < 0.8,
            data['volume_percentile'] * 0.5,  # Reduce during low volume
            data['volume_percentile']  # Normal weighting
        )
    )
    
    # Volatility Normalization
    # Calculate recent volatility (10-day std of returns)
    data['returns'] = data['close'].pct_change()
    data['volatility_10d'] = data['returns'].rolling(window=10, min_periods=5).std()
    
    # Avoid division by zero
    data['volatility_10d_adj'] = data['volatility_10d'].replace(0, np.nan)
    
    # Create Final Factor
    # Combine components with volume weighting and volatility normalization
    data['factor'] = (data['range_efficiency'] * data['volume_divergence'] * data['volume_weight']) / data['volatility_10d_adj']
    
    # Clean extreme values
    factor_series = data['factor'].copy()
    q_low = factor_series.quantile(0.01)
    q_high = factor_series.quantile(0.99)
    factor_series = np.where(factor_series < q_low, q_low, factor_series)
    factor_series = np.where(factor_series > q_high, q_high, factor_series)
    
    # Return as pandas Series with original index
    return pd.Series(factor_series, index=df.index, name='vol_adj_range_efficiency_volume')
