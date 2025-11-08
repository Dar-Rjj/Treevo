import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate regime-adaptive momentum factor with volume divergence analysis
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Calculate returns for volatility calculation
    daily_returns = data['close'].pct_change()
    
    # Multi-Timeframe Volatility-Adjusted Momentum
    # Short-term (3-day)
    data['ret_3d'] = data['close'].pct_change(3)
    data['vol_3d'] = daily_returns.rolling(window=3).std()
    
    # Medium-term (10-day)
    data['ret_10d'] = data['close'].pct_change(10)
    data['vol_10d'] = daily_returns.rolling(window=10).std()
    
    # Long-term (20-day)
    data['ret_20d'] = data['close'].pct_change(20)
    data['vol_20d'] = daily_returns.rolling(window=20).std()
    
    # Volume-Price Divergence Analysis
    data['volume_5d_slope'] = (data['volume'] - data['volume'].shift(5)) / data['volume'].shift(5)
    data['volume_10d_slope'] = (data['volume'] - data['volume'].shift(10)) / data['volume'].shift(10)
    data['ret_5d'] = data['close'].pct_change(5)
    
    # Divergence detection
    bearish_div = (data['ret_5d'] > 0) & (data['volume_5d_slope'] < 0)
    bullish_div = (data['ret_5d'] < 0) & (data['volume_5d_slope'] > 0)
    data['divergence_magnitude'] = abs(data['ret_5d']) * abs(data['volume_5d_slope'])
    
    # Volume confirmation strength
    data['volume_acceleration'] = data['volume_5d_slope'] / data['volume_10d_slope']
    data['volume_20d_avg'] = data['volume'].rolling(window=20).mean()
    data['sustained_volume'] = data['volume'].rolling(window=5).apply(
        lambda x: sum(x > data.loc[x.index[-1], 'volume_20d_avg']) if not pd.isna(data.loc[x.index[-1], 'volume_20d_avg']) else np.nan
    )
    
    # Market Regime Classification
    # Volatility regime
    high_vol = data['vol_10d'] > data['vol_20d']
    low_vol = data['vol_10d'] < (data['vol_20d'] * 0.7)
    normal_vol = ~high_vol & ~low_vol
    
    # Trend regime
    ma_5 = data['close'].rolling(window=5).mean()
    ma_10 = data['close'].rolling(window=10).mean()
    ma_20 = data['close'].rolling(window=20).mean()
    
    uptrend = (ma_10 > ma_20) & (ma_5 > ma_10)
    downtrend = (ma_10 < ma_20) & (ma_5 < ma_10)
    sideways = ~uptrend & ~downtrend
    
    # Volume regime
    high_volume = data['volume'] > (1.5 * data['volume_20d_avg'])
    low_volume = data['volume'] < (0.7 * data['volume_20d_avg'])
    normal_volume = ~high_volume & ~low_volume
    
    # Regime-Adaptive Weighting Scheme
    # Initialize weights
    weights_short = np.zeros(len(data))
    weights_medium = np.zeros(len(data))
    weights_long = np.zeros(len(data))
    
    # Volatility regime weights
    weights_short[high_vol] = 0.6
    weights_medium[high_vol] = 0.3
    weights_long[high_vol] = 0.1
    
    weights_short[low_vol] = 0.2
    weights_medium[low_vol] = 0.3
    weights_long[low_vol] = 0.5
    
    weights_short[normal_vol] = 0.33
    weights_medium[normal_vol] = 0.33
    weights_long[normal_vol] = 0.33
    
    # Trend regime adjustment
    trend_multiplier = np.ones(len(data))
    trend_multiplier[uptrend] = 1.2
    trend_multiplier[downtrend] = 0.8
    
    weights_short *= trend_multiplier
    weights_medium *= trend_multiplier
    weights_long *= trend_multiplier
    
    # Normalize weights to sum to 1
    total_weights = weights_short + weights_medium + weights_long
    weights_short = np.where(total_weights > 0, weights_short / total_weights, 0.33)
    weights_medium = np.where(total_weights > 0, weights_medium / total_weights, 0.33)
    weights_long = np.where(total_weights > 0, weights_long / total_weights, 0.33)
    
    # Volume regime multiplier
    volume_multiplier = np.ones(len(data))
    volume_multiplier[high_volume] = 1.2
    volume_multiplier[low_volume] = 0.8
    
    # Composite Alpha Factor Construction
    # Volatility-adjusted momentum components
    mom_short = data['ret_3d'] / data['vol_3d']
    mom_medium = data['ret_10d'] / data['vol_10d']
    mom_long = data['ret_20d'] / data['vol_20d']
    
    # Weighted momentum
    weighted_momentum = (weights_short * mom_short + 
                        weights_medium * mom_medium + 
                        weights_long * mom_long)
    
    # Volume divergence adjustment
    divergence_adjustment = np.zeros(len(data))
    divergence_adjustment[bearish_div] = -0.5 * data.loc[bearish_div, 'divergence_magnitude']
    divergence_adjustment[bullish_div] = 0.5 * data.loc[bullish_div, 'divergence_magnitude']
    
    # Final composite factor
    composite_factor = weighted_momentum + divergence_adjustment
    
    # Apply volume regime multiplier
    final_factor = composite_factor * volume_multiplier
    
    return pd.Series(final_factor, index=data.index)
