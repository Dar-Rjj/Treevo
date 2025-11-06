import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Compression-Acceleration Divergence with Liquidity Confirmation factor
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Scale Compression Detection
    # Volatility Compression Signals
    data['intraday_range'] = data['high'] - data['low']
    data['intraday_compression'] = data['intraday_range'] / data['intraday_range'].shift(1)
    
    data['close_change'] = abs(data['close'] - data['close'].shift(1))
    data['close_vol_compression'] = data['close_change'] / data['close_change'].shift(1)
    
    # Rolling volatility ratio (5-day)
    data['rolling_vol_5'] = data['close'].rolling(window=5).std()
    data['vol_ratio_5'] = data['rolling_vol_5'] / data['rolling_vol_5'].shift(5)
    
    # Compression Pattern Recognition
    # Consecutive compression days (intraday range < 80% of 5-day average)
    data['range_5d_avg'] = data['intraday_range'].rolling(window=5).mean()
    data['compression_flag'] = (data['intraday_range'] < 0.8 * data['range_5d_avg']).astype(int)
    data['consecutive_compression'] = data['compression_flag'] * (data['compression_flag'].groupby((data['compression_flag'] != data['compression_flag'].shift()).cumsum()).cumcount() + 1)
    
    # Compression depth vs 20-day historical volatility
    data['hist_vol_20'] = data['close'].rolling(window=20).std()
    data['compression_depth'] = data['intraday_range'] / data['hist_vol_20']
    
    # Acceleration-Decay Dynamics
    # Price Acceleration Component
    data['momentum_5d'] = data['close'] - data['close'].shift(5)
    data['momentum_change_5d'] = data['momentum_5d'] - data['momentum_5d'].shift(1)
    
    # 3-day momentum slope using linear regression
    def linear_slope_3d(series):
        if len(series) < 3:
            return np.nan
        x = np.arange(len(series))
        slope, _, _, _, _ = stats.linregress(x, series)
        return slope
    
    data['momentum_slope_3d'] = data['close'].rolling(window=3).apply(linear_slope_3d, raw=False)
    
    # Volume Acceleration Component
    data['volume_change_5d'] = data['volume'] - data['volume'].shift(5)
    data['volume_slope_3d'] = data['volume'].rolling(window=3).apply(linear_slope_3d, raw=False)
    
    # Decay Pattern Analysis
    data['price_decay'] = (data['close'].shift(1) - data['open']) / data['close'].shift(1)
    data['intraday_decay'] = (data['high'] - data['close']) / (data['high'] - data['low']).replace(0, np.nan)
    data['volume_persistence'] = data['volume'] / data['volume'].shift(1)
    
    # Compression-Acceleration Divergence
    # Normalize components for divergence calculation
    compression_components = ['intraday_compression', 'close_vol_compression', 'vol_ratio_5', 'compression_depth']
    acceleration_components = ['momentum_change_5d', 'momentum_slope_3d', 'volume_change_5d', 'volume_slope_3d']
    
    # Z-score normalization
    for col in compression_components + acceleration_components:
        if col in data.columns:
            mean_val = data[col].rolling(window=20, min_periods=10).mean()
            std_val = data[col].rolling(window=20, min_periods=10).std()
            data[f'{col}_z'] = (data[col] - mean_val) / std_val.replace(0, np.nan)
    
    # Compression score (inverse of z-scores since lower values indicate compression)
    compression_score = -(data['intraday_compression_z'] + data['close_vol_compression_z'] + 
                         data['vol_ratio_5_z'] + data['compression_depth_z']) / 4
    
    # Acceleration score
    acceleration_score = (data['momentum_change_5d_z'] + data['momentum_slope_3d_z'] + 
                         data['volume_change_5d_z'] + data['volume_slope_3d_z']) / 4
    
    # Divergence score (compression * acceleration)
    data['divergence_score'] = compression_score * acceleration_score
    
    # Divergence intensity with duration weighting
    data['divergence_intensity'] = data['divergence_score'] * np.sqrt(data['consecutive_compression'].fillna(0))
    
    # Liquidity Confirmation Framework
    # Directional Liquidity Response
    data['upside_liquidity'] = np.where(data['close'] > data['close'].shift(1), data['volume'], 0)
    data['downside_liquidity'] = np.where(data['close'] < data['close'].shift(1), data['volume'], 0)
    
    upside_volume_5d = data['upside_liquidity'].rolling(window=5).sum()
    downside_volume_5d = data['downside_liquidity'].rolling(window=5).sum()
    data['liquidity_asymmetry'] = upside_volume_5d / downside_volume_5d.replace(0, np.nan)
    
    # Volume-Liquidity Integration
    data['volume_5d_avg'] = data['volume'].rolling(window=5).mean()
    data['volume_surge'] = data['volume'] / data['volume_5d_avg']
    data['liquidity_measure'] = data['amount'] / data['volume'].replace(0, np.nan)
    
    # Combined volume-liquidity signal
    data['volume_liquidity_signal'] = data['volume_surge'] * data['liquidity_measure']
    
    # Liquidity confirmation strength
    data['liquidity_confirmation'] = (data['liquidity_asymmetry'].fillna(1) * 
                                     data['volume_liquidity_signal'].fillna(1))
    
    # Hierarchical Factor Construction
    # Core divergence components
    divergence_components = data['divergence_intensity'].fillna(0)
    
    # Adaptive signal generation with volatility scaling
    data['daily_volatility'] = (data['high'] - data['low']) / data['close']
    vol_scale = 1 / data['daily_volatility'].rolling(window=10).mean().replace(0, np.nan)
    
    # Final alpha signal: Compression-acceleration divergence × liquidity confirmation × volatility scaling
    alpha_signal = (divergence_components * 
                   data['liquidity_confirmation'].fillna(1) * 
                   vol_scale.fillna(1))
    
    # Apply regime-based weighting
    divergence_strength = data['divergence_score'].abs().rolling(window=10).mean()
    confirmation_strength = data['liquidity_confirmation'].abs().rolling(window=10).mean()
    
    # Aggressive weighting for high divergence + strong confirmation
    regime_multiplier = np.where(
        (divergence_strength > divergence_strength.quantile(0.7)) & 
        (confirmation_strength > confirmation_strength.quantile(0.7)),
        1.5,  # Aggressive
        np.where(
            (divergence_strength < divergence_strength.quantile(0.3)) & 
            (confirmation_strength < confirmation_strength.quantile(0.3)),
            0.5,  # Defensive
            1.0   # Neutral
        )
    )
    
    final_alpha = alpha_signal * regime_multiplier
    
    return final_alpha
