import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Price-Volume Asymmetry Dynamics
    # Calculate returns
    data['returns'] = data['close'].pct_change()
    
    # Up/Down Volume Pressure
    data['up_day'] = (data['returns'] > 0).astype(int)
    data['down_day'] = (data['returns'] < 0).astype(int)
    
    # Rolling up/down volume sums over 5 days
    data['up_volume_5d'] = (data['up_day'] * data['volume']).rolling(window=5, min_periods=3).sum()
    data['down_volume_5d'] = (data['down_day'] * data['volume']).rolling(window=5, min_periods=3).sum()
    data['total_volume_5d'] = data['volume'].rolling(window=5, min_periods=3).sum()
    
    # Volume pressure
    data['volume_pressure'] = (data['up_volume_5d'] - data['down_volume_5d']) / data['total_volume_5d']
    
    # Asymmetric Momentum
    data['high_5d'] = data['high'].rolling(window=5, min_periods=3).max()
    data['low_5d'] = data['low'].rolling(window=5, min_periods=3).min()
    
    data['high_momentum'] = (data['high'] - data['high_5d']) / data['high_5d']
    data['low_momentum'] = (data['low'] - data['low_5d']) / data['low_5d']
    
    data['asymmetric_momentum'] = (data['high_momentum'] - data['low_momentum']) * data['volume_pressure']
    
    # Regime-Switching Volatility Clustering
    data['high_low_range'] = (data['high'] - data['low']) / data['close']
    data['volatility_20d'] = data['high_low_range'].rolling(window=20, min_periods=10).mean()
    data['volatility_50d'] = data['high_low_range'].rolling(window=50, min_periods=25).mean()
    
    # Volatility regime ratio
    data['volatility_regime'] = data['volatility_20d'] / data['volatility_50d']
    
    # 5-day return
    data['return_5d'] = data['close'].pct_change(5)
    
    # Regime-adaptive momentum
    data['regime_adaptive_momentum'] = data['return_5d'] * data['volatility_regime']
    
    # Volume regime confirmation
    volatility_threshold = data['volatility_20d'].rolling(window=50, min_periods=25).median()
    data['high_vol_regime'] = (data['volatility_20d'] > volatility_threshold).astype(int)
    data['low_vol_regime'] = (data['volatility_20d'] <= volatility_threshold).astype(int)
    
    # Calculate regime-specific average volumes
    high_vol_mask = data['high_vol_regime'] == 1
    low_vol_mask = data['low_vol_regime'] == 1
    
    data['high_vol_avg_volume'] = data['volume'].rolling(window=50, min_periods=25).apply(
        lambda x: x[high_vol_mask.iloc[-len(x):].values].mean() if high_vol_mask.iloc[-len(x):].sum() > 0 else np.nan, 
        raw=False
    )
    
    data['low_vol_avg_volume'] = data['volume'].rolling(window=50, min_periods=25).apply(
        lambda x: x[low_vol_mask.iloc[-len(x):].values].mean() if low_vol_mask.iloc[-len(x):].sum() > 0 else np.nan, 
        raw=False
    )
    
    # Volume regime confirmation
    data['volume_regime_confirmation'] = np.where(
        data['high_vol_regime'] == 1,
        data['volume'] / data['high_vol_avg_volume'],
        data['volume'] / data['low_vol_avg_volume']
    )
    
    # Price-Level Dependent Reversal
    data['high_20d'] = data['high'].rolling(window=20, min_periods=10).max()
    data['low_20d'] = data['low'].rolling(window=20, min_periods=10).min()
    
    # Relative price position
    data['relative_position'] = (data['close'] - data['low_20d']) / (data['high_20d'] - data['low_20d'])
    
    # Position-based reversal
    data['position_reversal'] = -np.sign(data['return_5d']) * data['relative_position']
    
    # Volume intensity
    high_pos_mask = data['relative_position'] > 0.8
    low_pos_mask = data['relative_position'] < 0.2
    
    data['high_pos_avg_volume'] = data['volume'].rolling(window=50, min_periods=25).apply(
        lambda x: x[high_pos_mask.iloc[-len(x):].values].mean() if high_pos_mask.iloc[-len(x):].sum() > 0 else np.nan, 
        raw=False
    )
    
    data['low_pos_avg_volume'] = data['volume'].rolling(window=50, min_periods=25).apply(
        lambda x: x[low_pos_mask.iloc[-len(x):].values].mean() if low_pos_mask.iloc[-len(x):].sum() > 0 else np.nan, 
        raw=False
    )
    
    data['volume_intensity'] = np.where(
        high_pos_mask,
        data['volume'] / data['high_pos_avg_volume'],
        np.where(
            low_pos_mask,
            data['volume'] / data['low_pos_avg_volume'],
            data['volume'] / data['volume'].rolling(window=20, min_periods=10).mean()
        )
    )
    
    # Intraday Pattern Persistence
    data['open_to_high'] = (data['high'] - data['open']) / data['open']
    data['low_to_close'] = (data['close'] - data['low']) / data['low']
    data['morning_vs_afternoon'] = data['open_to_high'] - data['low_to_close']
    
    # Pattern consistency (3-day average pattern similarity)
    data['pattern_3d_avg'] = data['morning_vs_afternoon'].rolling(window=3, min_periods=2).mean()
    data['pattern_consistency'] = 1 - abs(data['morning_vs_afternoon'] - data['pattern_3d_avg'])
    
    # Volume pattern confirmation
    pattern_mask = abs(data['morning_vs_afternoon']) > data['morning_vs_afternoon'].rolling(window=20, min_periods=10).std()
    
    data['pattern_day_avg_volume'] = data['volume'].rolling(window=50, min_periods=25).apply(
        lambda x: x[pattern_mask.iloc[-len(x):].values].mean() if pattern_mask.iloc[-len(x):].sum() > 0 else np.nan, 
        raw=False
    )
    
    data['non_pattern_avg_volume'] = data['volume'].rolling(window=50, min_periods=25).apply(
        lambda x: x[~pattern_mask.iloc[-len(x):].values].mean() if (~pattern_mask.iloc[-len(x):]).sum() > 0 else np.nan, 
        raw=False
    )
    
    data['volume_pattern_confirmation'] = np.where(
        pattern_mask,
        data['volume'] / data['pattern_day_avg_volume'],
        data['volume'] / data['non_pattern_avg_volume']
    )
    
    # Liquidity-Driven Momentum Acceleration
    data['volume_5d_change'] = data['volume'].pct_change(5)
    data['volume_20d_change'] = data['volume'].pct_change(20)
    data['liquidity_momentum'] = data['volume_5d_change'] / data['volume_20d_change']
    
    # Price acceleration
    data['return_3d'] = data['close'].pct_change(3)
    data['return_8d'] = data['close'].pct_change(8)
    data['price_acceleration'] = (data['return_3d'] - data['return_8d']) * data['liquidity_momentum']
    
    # Liquidity regime filter
    liquidity_median = data['liquidity_momentum'].rolling(window=50, min_periods=25).median()
    data['high_liquidity_regime'] = (data['liquidity_momentum'] > liquidity_median).astype(int)
    
    # Combine all components with appropriate weights
    # Use liquidity regime filter for momentum components
    data['filtered_momentum'] = np.where(
        data['high_liquidity_regime'] == 1,
        data['price_acceleration'],
        0
    )
    
    # Final factor combination
    factor = (
        0.25 * data['asymmetric_momentum'] +
        0.20 * data['regime_adaptive_momentum'] +
        0.15 * data['volume_regime_confirmation'] +
        0.15 * data['position_reversal'] +
        0.10 * data['volume_intensity'] +
        0.10 * data['pattern_consistency'] +
        0.05 * data['volume_pattern_confirmation'] +
        0.10 * data['filtered_momentum']
    )
    
    # Fill NaN values with 0
    factor = factor.fillna(0)
    
    return factor
