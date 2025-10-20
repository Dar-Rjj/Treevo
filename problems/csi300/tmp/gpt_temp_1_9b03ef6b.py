import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volume-Weighted Price Reversal with Volatility Regime Filtering
    Generates alpha factor based on price reversal signals confirmed by volume
    and filtered by volatility regime
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate Price Reversal Signals
    # Short-term reversal components
    data['return_1d'] = data['close'].pct_change(1)
    data['return_2d'] = data['close'].pct_change(2)
    data['return_3d'] = data['close'].pct_change(3)
    
    # Derive Reversal Strength Metrics
    data['return_decay'] = np.where(
        data['return_3d'] != 0, 
        data['return_1d'] / data['return_3d'], 
        0
    )
    
    # Return consistency (sign agreement across 1,2,3-day returns)
    data['sign_consistency'] = (
        np.sign(data['return_1d']) * np.sign(data['return_2d']) * 
        np.sign(data['return_3d'])
    )
    
    # Magnitude ratio (absolute 1-day return vs 3-day return)
    data['magnitude_ratio'] = np.where(
        abs(data['return_3d']) > 0,
        abs(data['return_1d']) / abs(data['return_3d']),
        0
    )
    
    # Analyze Volume-Based Confirmation
    # Volume surge indicators
    data['vol_ma_5d'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_surge'] = np.where(
        data['vol_ma_5d'] > 0,
        data['volume'] / data['vol_ma_5d'],
        1
    )
    
    data['volume_acceleration'] = data['volume'] / data['volume'].shift(1)
    data['volume_acceleration'] = data['volume_acceleration'].fillna(1)
    
    # Volume persistence (3-day volume trend direction)
    data['volume_trend'] = (
        data['volume'].rolling(window=3, min_periods=2).apply(
            lambda x: 1 if len(x) >= 2 and x.iloc[-1] > x.iloc[0] else -1, 
            raw=False
        )
    )
    
    # Calculate Volume-Price Divergence
    data['volume_price_divergence'] = (
        data['volume_surge'] * (1 - abs(data['return_1d']))
    )
    
    # Incorporate Amount Data for Order Flow Context
    data['amount_ma_5d'] = data['amount'].rolling(window=5, min_periods=3).mean()
    data['amount_surge'] = np.where(
        data['amount_ma_5d'] > 0,
        data['amount'] / data['amount_ma_5d'],
        1
    )
    
    data['amount_to_volume_ratio'] = np.where(
        data['volume'] > 0,
        data['amount'] / data['volume'],
        0
    )
    data['amount_volume_ratio_change'] = (
        data['amount_to_volume_ratio'] / 
        data['amount_to_volume_ratio'].shift(1)
    )
    data['amount_volume_ratio_change'] = data['amount_volume_ratio_change'].fillna(1)
    
    # Large order detection (amount spikes)
    data['amount_spike'] = np.where(
        data['amount_surge'] > 1.5,
        data['amount_surge'],
        1
    )
    
    # Assess Volatility Environment
    # Volatility regime indicators
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['range_ma_20d'] = data['daily_range'].rolling(window=20, min_periods=10).mean()
    data['range_ratio'] = np.where(
        data['range_ma_20d'] > 0,
        data['daily_range'] / data['range_ma_20d'],
        1
    )
    
    data['abs_return'] = abs(data['return_1d'])
    data['abs_return_ma_10d'] = data['abs_return'].rolling(window=10, min_periods=5).mean()
    data['return_vol_ratio'] = np.where(
        data['abs_return_ma_10d'] > 0,
        data['abs_return'] / data['abs_return_ma_10d'],
        1
    )
    
    # Volatility clustering indicator
    data['volatility_clustering'] = (
        data['abs_return'].rolling(window=5, min_periods=3).std() / 
        data['abs_return'].rolling(window=20, min_periods=10).std()
    )
    data['volatility_clustering'] = data['volatility_clustering'].fillna(1)
    
    # Classify Market Conditions
    data['volatility_regime'] = np.where(
        data['range_ratio'] > 1.2, 'high',
        np.where(data['range_ratio'] < 0.8, 'low', 'normal')
    )
    
    # Compute Base Reversal Signal
    # Short-term return decay × Volume surge strength
    base_reversal = (
        data['return_decay'] * data['volume_surge'] * 
        data['amount_volume_ratio_change']
    )
    
    # Adjust for multi-timeframe consistency
    consistency_adjustment = np.where(
        data['sign_consistency'] == -1,  # Reversal pattern
        1.2,  # Enhance true reversals
        0.8   # Reduce continuation patterns
    )
    
    base_reversal = base_reversal * consistency_adjustment
    
    # Apply Volatility Regime Filtering
    regime_multiplier = np.where(
        data['volatility_regime'] == 'high', 1.4,
        np.where(data['volatility_regime'] == 'low', 0.8, 1.0)
    )
    
    # Final alpha factor
    alpha_factor = base_reversal * regime_multiplier
    
    # Additional filtering based on volume confirmation
    volume_confirmation = np.where(
        (data['volume_surge'] > 1.1) & (data['amount_spike'] > 1.2),
        1.1,  # Strong confirmation
        np.where(
            (data['volume_surge'] > 0.9) & (data['amount_spike'] > 1.0),
            1.0,  # Moderate confirmation
            0.8   # Weak confirmation
        )
    )
    
    alpha_factor = alpha_factor * volume_confirmation
    
    # Ensure no lookahead bias and return clean series
    result = pd.Series(alpha_factor, index=data.index)
    result = result.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return result
