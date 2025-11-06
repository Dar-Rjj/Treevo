import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Volatility-Adaptive Acceleration Divergence factor
    """
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Multi-Timeframe Volatility Assessment
    # Short-term volatility (1-5 days)
    data['short_vol'] = (data['high'] - data['low']).rolling(window=3).std()
    data['short_vol_accel'] = data['short_vol'].diff(2) / data['short_vol'].shift(2)
    
    # Medium-term volatility (5-15 days)
    data['medium_vol'] = (data['high'] - data['low']).rolling(window=10).std()
    data['medium_vol_trend'] = data['medium_vol'].rolling(window=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True
    )
    
    # Long-term volatility (15-30 days)
    data['long_vol'] = (data['high'] - data['low']).rolling(window=20).std()
    data['vol_cycle'] = data['long_vol'] - data['long_vol'].rolling(window=10).mean()
    
    # Adaptive Regime Classification
    vol_regime_threshold_high = data['medium_vol'].quantile(0.7)
    vol_regime_threshold_low = data['medium_vol'].quantile(0.3)
    
    data['vol_regime'] = 1  # Normal volatility
    data.loc[data['medium_vol'] > vol_regime_threshold_high, 'vol_regime'] = 2  # High volatility
    data.loc[data['medium_vol'] < vol_regime_threshold_low, 'vol_regime'] = 0  # Low volatility
    
    # Multi-Timeframe Price Acceleration Analysis
    # Short-term price acceleration (1-3 days)
    data['price_momentum_short'] = data['close'].pct_change(periods=2)
    data['price_accel_short'] = data['price_momentum_short'].diff()
    
    # Medium-term price acceleration (5-10 days)
    data['price_momentum_medium'] = data['close'].pct_change(periods=7)
    data['price_accel_medium'] = data['price_momentum_medium'].rolling(window=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 2)[2] if len(x) == 5 else np.nan, raw=True
    )
    
    # Long-term price acceleration (20-40 days)
    data['price_trend_long'] = data['close'].rolling(window=30).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True
    )
    data['price_accel_long'] = data['price_trend_long'].diff(5)
    
    # Multi-Timeframe Volume Acceleration Analysis
    # Short-term volume acceleration (1-3 days)
    data['volume_momentum_short'] = data['volume'].pct_change(periods=2)
    data['volume_accel_short'] = data['volume_momentum_short'].diff()
    
    # Medium-term volume acceleration (5-10 days)
    data['volume_trend_medium'] = data['volume'].rolling(window=8).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True
    )
    data['volume_accel_medium'] = data['volume_trend_medium'].diff(3)
    
    # Long-term volume cycle (20-40 days)
    data['volume_cycle_long'] = data['volume'] - data['volume'].rolling(window=25).mean()
    data['volume_cycle_accel'] = data['volume_cycle_long'].diff(5)
    
    # Cross-Timeframe Acceleration Alignment
    data['accel_alignment_short'] = (
        np.sign(data['price_accel_short']) * np.sign(data['volume_accel_short'])
    )
    data['accel_alignment_medium'] = (
        np.sign(data['price_accel_medium']) * np.sign(data['volume_accel_medium'])
    )
    data['accel_alignment_long'] = (
        np.sign(data['price_accel_long']) * np.sign(data['volume_cycle_accel'])
    )
    
    # Volatility-Adaptive Divergence Detection
    # Price-Volume Acceleration Divergence
    data['pv_divergence_short'] = (
        data['price_accel_short'] - data['volume_accel_short'].rolling(window=3).mean()
    )
    data['pv_divergence_medium'] = (
        data['price_accel_medium'] - data['volume_accel_medium'].rolling(window=5).mean()
    )
    data['pv_divergence_long'] = (
        data['price_accel_long'] - data['volume_cycle_accel'].rolling(window=10).mean()
    )
    
    # Multi-Timeframe Divergence Consistency
    data['divergence_consistency'] = (
        np.sign(data['pv_divergence_short']) + 
        np.sign(data['pv_divergence_medium']) + 
        np.sign(data['pv_divergence_long'])
    )
    
    # Adaptive Composite Factor Generation
    # Volatility-Regime Weighted Signals
    def regime_weighted_signal(row):
        if row['vol_regime'] == 2:  # High volatility - focus on reversals
            return -row['pv_divergence_short'] * 0.6 - row['pv_divergence_medium'] * 0.4
        elif row['vol_regime'] == 0:  # Low volatility - focus on breakouts
            return row['pv_divergence_short'] * 0.7 + row['pv_divergence_medium'] * 0.3
        else:  # Normal volatility - balanced approach
            return (
                row['pv_divergence_short'] * 0.4 + 
                row['pv_divergence_medium'] * 0.4 + 
                row['pv_divergence_long'] * 0.2
            )
    
    data['regime_weighted_signal'] = data.apply(regime_weighted_signal, axis=1)
    
    # Multi-Timeframe Acceleration Integration
    data['acceleration_integration'] = (
        data['price_accel_short'].fillna(0) * 0.3 +
        data['price_accel_medium'].fillna(0) * 0.4 +
        data['price_accel_long'].fillna(0) * 0.3
    )
    
    # Final Adaptive Factor Components
    volatility_regime_score = data['vol_regime'].map({0: 1.2, 1: 1.0, 2: 0.8})
    
    cross_timeframe_divergence = (
        data['pv_divergence_short'].fillna(0) * 0.4 +
        data['pv_divergence_medium'].fillna(0) * 0.35 +
        data['pv_divergence_long'].fillna(0) * 0.25
    )
    
    volume_accel_confirmation = (
        data['volume_accel_short'].fillna(0) * 0.5 +
        data['volume_accel_medium'].fillna(0) * 0.3 +
        data['volume_cycle_accel'].fillna(0) * 0.2
    )
    
    # Final Composite Factor
    factor = (
        volatility_regime_score * 
        cross_timeframe_divergence * 
        (1 + 0.2 * np.sign(volume_accel_confirmation)) *
        data['divergence_consistency'].fillna(0)
    )
    
    # Normalize the factor
    factor = (factor - factor.rolling(window=50).mean()) / factor.rolling(window=50).std()
    
    return factor
