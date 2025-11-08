import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate a novel alpha factor combining volatility-scaled momentum acceleration,
    volume-price divergence, intraday pattern confirmation, and regime-adaptive signals.
    """
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Volatility-Scaled Momentum Acceleration
    # Multi-Timeframe Momentum
    data['mom_3d'] = data['close'] / data['close'].shift(3) - 1
    data['mom_5d'] = data['close'] / data['close'].shift(5) - 1
    data['mom_10d'] = data['close'] / data['close'].shift(10) - 1
    
    # Momentum Acceleration
    data['accel_3_5'] = data['mom_5d'] - data['mom_3d']
    data['accel_5_10'] = data['mom_10d'] - data['mom_5d']
    data['net_accel'] = (data['accel_3_5'] + data['accel_5_10']) / 2
    
    # Volatility Scaling
    data['daily_ret'] = data['close'].pct_change()
    data['vol_3d'] = data['daily_ret'].rolling(window=3).std()
    data['vol_5d'] = data['daily_ret'].rolling(window=5).std()
    data['vol_10d'] = data['daily_ret'].rolling(window=10).std()
    
    # Combined volatility-scaled momentum factor
    vol_geomean = np.exp((np.log(data['vol_3d'].replace(0, np.nan)) + 
                         np.log(data['vol_5d'].replace(0, np.nan)) + 
                         np.log(data['vol_10d'].replace(0, np.nan))) / 3)
    data['vol_scaled_mom'] = data['net_accel'] / vol_geomean
    
    # Volume-Price Divergence
    # Price Strength Components
    data['intraday_strength'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['high_low_efficiency'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    data['price_persistence'] = (data['close'] - data['close'].shift(1)) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Core price strength (geometric mean)
    price_components = [data['intraday_strength'], data['high_low_efficiency'], data['price_persistence']]
    valid_mask = ~pd.concat(price_components, axis=1).isna().any(axis=1)
    data['core_price_strength'] = np.nan
    data.loc[valid_mask, 'core_price_strength'] = np.exp(
        (np.log(data.loc[valid_mask, 'intraday_strength'].abs()) + 
         np.log(data.loc[valid_mask, 'high_low_efficiency'].abs()) + 
         np.log(data.loc[valid_mask, 'price_persistence'].abs())) / 3
    ) * np.sign(data.loc[valid_mask, 'intraday_strength'])
    
    # Volume Divergence Signals
    data['vol_3d_ratio'] = data['volume'] / np.exp(data['volume'].shift(1).rolling(window=2).apply(lambda x: np.log(x).mean()))
    data['vol_5d_ratio'] = data['volume'] / np.exp(data['volume'].shift(1).rolling(window=4).apply(lambda x: np.log(x).mean()))
    data['vol_trend_div'] = data['vol_3d_ratio'] - data['vol_5d_ratio']
    
    # Combined volume-price divergence factor
    data['vol_price_div'] = data['core_price_strength'] * data['vol_trend_div']
    
    # Intraday Pattern Confirmation
    # Morning Session Signals
    data['opening_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['morning_efficiency'] = (data['high'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['morning_support'] = (data['open'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Afternoon Session Signals
    data['afternoon_momentum'] = (data['close'] - data['high']) / (data['high'] - data['low']).replace(0, np.nan)
    data['afternoon_support'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    data['closing_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Intraday Consistency
    data['session_alignment'] = (data['morning_efficiency'] - data['morning_support']) * (data['afternoon_momentum'] - data['afternoon_support'])
    data['pattern_factor'] = data['session_alignment'] * data['closing_efficiency']
    
    # Regime-Adaptive Signal Integration
    # Short-term Framework (1-3 days)
    data['short_price_mom'] = data['close'] / data['close'].shift(2) - 1
    data['short_vol_accel'] = data['volume'] / np.exp(data['volume'].shift(1).rolling(window=2).apply(lambda x: np.log(x).mean()))
    data['short_range_eff'] = (data['close'] - data['close'].shift(1)) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Medium-term Framework (5-10 days)
    data['medium_price_trend'] = data['close'] / data['close'].shift(7) - 1
    data['medium_vol_trend'] = data['volume'] / np.exp(data['volume'].shift(1).rolling(window=6).apply(lambda x: np.log(x).mean()))
    
    # Daily ranges for volatility regime
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['vol_regime'] = data['daily_range'] / np.exp(data['daily_range'].shift(1).rolling(window=6).apply(lambda x: np.log(x).mean()))
    
    # Adaptive Combination
    # Signal convergence (geometric mean of price signals)
    price_signals = [data['short_price_mom'], data['medium_price_trend']]
    valid_price = ~pd.concat(price_signals, axis=1).isna().any(axis=1)
    data['signal_convergence'] = np.nan
    data.loc[valid_price, 'signal_convergence'] = np.exp(
        (np.log(data.loc[valid_price, 'short_price_mom'].abs()) + 
         np.log(data.loc[valid_price, 'medium_price_trend'].abs())) / 2
    ) * np.sign(data.loc[valid_price, 'short_price_mom'])
    
    # Volume confirmation (geometric mean of volume signals)
    vol_signals = [data['short_vol_accel'], data['medium_vol_trend']]
    valid_vol = ~pd.concat(vol_signals, axis=1).isna().any(axis=1)
    data['volume_confirmation'] = np.nan
    data.loc[valid_vol, 'volume_confirmation'] = np.exp(
        (np.log(data.loc[valid_vol, 'short_vol_accel']) + 
         np.log(data.loc[valid_vol, 'medium_vol_trend'])) / 2
    )
    
    # Regime-adaptive factor
    data['regime_adaptive'] = (data['signal_convergence'] * data['volume_confirmation']) / data['vol_regime'].replace(0, np.nan)
    
    # Final alpha factor - weighted combination of all components
    components = ['vol_scaled_mom', 'vol_price_div', 'pattern_factor', 'regime_adaptive']
    weights = [0.3, 0.25, 0.2, 0.25]  # Equal weighting for simplicity
    
    # Normalize each component by its rolling z-score
    final_factor = pd.Series(0, index=data.index)
    for i, comp in enumerate(components):
        comp_data = data[comp].copy()
        # Remove outliers using rolling z-score
        rolling_mean = comp_data.rolling(window=20, min_periods=10).mean()
        rolling_std = comp_data.rolling(window=20, min_periods=10).std()
        z_score = (comp_data - rolling_mean) / rolling_std.replace(0, np.nan)
        # Winsorize at ±3 standard deviations
        comp_clean = comp_data.copy()
        comp_clean[z_score.abs() > 3] = np.nan
        # Normalize and weight
        comp_norm = (comp_clean - comp_clean.rolling(window=20, min_periods=10).mean()) / comp_clean.rolling(window=20, min_periods=10).std()
        final_factor += weights[i] * comp_norm
    
    return final_factor
