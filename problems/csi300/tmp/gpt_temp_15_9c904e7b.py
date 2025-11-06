import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Compression-Acceleration Divergence with Liquidity Confirmation alpha factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Scale Compression Detection
    # Volatility Compression Signals
    data['intraday_range'] = data['high'] - data['low']
    data['intraday_compression'] = data['intraday_range'] / data['intraday_range'].shift(1)
    
    data['close_change'] = abs(data['close'] - data['close'].shift(1))
    data['close_vol_compression'] = data['close_change'] / data['close_change'].shift(1)
    
    # Rolling volatility ratio (5-day)
    data['close_std_5'] = data['close'].rolling(window=5).std()
    data['rolling_vol_ratio'] = data['close_std_5'] / data['close_std_5'].shift(5)
    
    # Compression Pattern Recognition
    # Consecutive compression days count
    compression_condition = (
        (data['intraday_compression'] < 0.8) & 
        (data['close_vol_compression'] < 0.8) & 
        (data['rolling_vol_ratio'] < 0.9)
    )
    data['compression_count'] = compression_condition.rolling(window=10, min_periods=1).sum()
    
    # Compression depth vs historical volatility
    data['hist_vol_20'] = data['close'].pct_change().rolling(window=20).std()
    data['compression_depth'] = (1 - data['intraday_compression']) * data['hist_vol_20']
    
    # Acceleration-Decay Dynamics
    # Price Acceleration Component
    data['momentum_5'] = data['close'] - data['close'].shift(5)
    data['momentum_change'] = data['momentum_5'] - data['momentum_5'].shift(1)
    
    # 3-day momentum slope using linear regression
    def linear_slope_3(series):
        if len(series) < 3:
            return np.nan
        x = np.arange(len(series))
        slope, _, _, _, _ = stats.linregress(x, series)
        return slope
    
    data['price_slope_3'] = data['close'].rolling(window=3).apply(linear_slope_3, raw=False)
    
    # Volume Acceleration Component
    data['volume_change_5'] = data['volume'] - data['volume'].shift(5)
    data['volume_slope_3'] = data['volume'].rolling(window=3).apply(linear_slope_3, raw=False)
    
    # Decay Pattern Analysis
    data['price_decay'] = (data['close'].shift(1) - data['open']) / data['close'].shift(1)
    data['intraday_decay'] = (data['high'] - data['close']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Compression-Acceleration Divergence
    # Normalize components for divergence calculation
    data['compression_score'] = (
        (1 - data['intraday_compression'].clip(0, 2)) + 
        (1 - data['close_vol_compression'].clip(0, 2)) + 
        (1 - data['rolling_vol_ratio'].clip(0, 2))
    ) / 3
    
    data['acceleration_score'] = (
        data['momentum_change'].fillna(0) / data['close'].rolling(window=10).std().replace(0, np.nan) +
        data['price_slope_3'].fillna(0) / data['close'].rolling(window=10).std().replace(0, np.nan) +
        data['volume_slope_3'].fillna(0) / data['volume'].rolling(window=10).std().replace(0, np.nan)
    ) / 3
    
    # Divergence Signal Generation
    data['divergence_raw'] = data['compression_score'] * data['acceleration_score']
    
    # Divergence Intensity Quantification
    data['divergence_intensity'] = (
        data['compression_score'].abs() * data['acceleration_score'].abs() * 
        (1 + data['compression_count'] / 10)
    )
    
    # Liquidity Confirmation Framework
    # Directional Liquidity Response
    up_condition = data['close'] > data['close'].shift(1)
    down_condition = data['close'] < data['close'].shift(1)
    
    data['upside_volume'] = np.where(up_condition, data['volume'], 0)
    data['downside_volume'] = np.where(down_condition, data['volume'], 0)
    
    data['liquidity_asymmetry'] = (
        data['upside_volume'].rolling(window=5).mean() / 
        data['downside_volume'].rolling(window=5).mean().replace(0, np.nan)
    )
    
    # Volume-Liquidity Integration
    data['volume_ma_5'] = data['volume'].rolling(window=5).mean()
    data['volume_surge'] = data['volume'] / data['volume_ma_5'].replace(0, np.nan)
    
    data['liquidity_measure'] = data['amount'] / data['volume'].replace(0, np.nan)
    
    # Liquidity Confirmation Timing
    data['liquidity_response'] = (
        (data['volume_surge'] * np.sign(data['divergence_raw'])) + 
        (data['liquidity_asymmetry'] - 1) * np.sign(data['divergence_raw'])
    )
    
    # Hierarchical Factor Construction
    # Core Divergence Components
    data['divergence_score'] = data['divergence_raw'] * data['divergence_intensity']
    
    # Liquidity Confirmation Integration
    data['liquidity_confirmation'] = (
        data['liquidity_response'].fillna(0) * 
        data['liquidity_measure'] / data['liquidity_measure'].rolling(window=20).mean()
    )
    
    # Adaptive Signal Generation
    # Regime-based multiplier
    high_divergence = data['divergence_intensity'] > data['divergence_intensity'].rolling(window=20).quantile(0.7)
    strong_confirmation = data['liquidity_confirmation'].abs() > data['liquidity_confirmation'].abs().rolling(window=20).quantile(0.7)
    
    data['regime_multiplier'] = np.where(
        high_divergence & strong_confirmation, 1.5,
        np.where(
            (~high_divergence) & (~strong_confirmation), 0.5,
            1.0
        )
    )
    
    # Volatility scaling
    data['daily_volatility'] = (data['high'] - data['low']) / data['close']
    data['volatility_scale'] = 1 / (data['daily_volatility'].rolling(window=20).mean().replace(0, np.nan))
    
    # Final Alpha Signal
    data['alpha_signal'] = (
        data['divergence_score'] * 
        data['liquidity_confirmation'] * 
        data['regime_multiplier'] * 
        data['volatility_scale']
    )
    
    # Clean up and return
    result = data['alpha_signal'].fillna(0)
    return result
