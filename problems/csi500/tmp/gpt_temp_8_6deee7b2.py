import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Generate a novel alpha factor combining multiple market dynamics:
    - Volatility-adjusted multi-timeframe momentum
    - Volume-price efficiency convergence
    - Session-based price dynamics
    - Multi-timeframe regime detection
    """
    df = data.copy()
    
    # Volatility-Adjusted Multi-Timeframe Momentum
    # Momentum components
    df['ultra_short_mom'] = df['close'] / df['close'].shift(1) - 1
    df['short_term_mom'] = df['close'] / df['close'].shift(3) - 1
    df['medium_term_mom'] = df['close'] / df['close'].shift(8) - 1
    
    # Volatility estimation
    df['ultra_short_vol'] = df['close'].pct_change().rolling(window=3).std()
    df['short_term_vol'] = df['close'].pct_change().rolling(window=5).std()
    df['medium_term_vol'] = df['close'].pct_change().rolling(window=8).std()
    
    # Signal construction
    df['geometric_mom'] = np.sign(df['ultra_short_mom'] * df['short_term_mom'] * df['medium_term_mom']) * \
                         np.power(np.abs(df['ultra_short_mom'] * df['short_term_mom'] * df['medium_term_mom']), 1/3)
    df['vol_weighted_mom'] = df['geometric_mom'] / (df['ultra_short_vol'] + df['short_term_vol'] + df['medium_term_vol'])
    
    # Volume-Price Efficiency Convergence
    # Price efficiency metrics
    df['intraday_eff'] = (df['close'] - df['open']) / (df['high'] - df['low'])
    df['high_low_pos'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    df['gap_eff'] = (df['open'] - df['close'].shift(1)) / (df['high'] - df['low'])
    
    # Volume quality signals
    df['vol_persistence'] = df['volume'] / df['volume'].shift(1)
    df['vol_acceleration'] = (df['volume'] / df['volume'].shift(1)) / (df['volume'].shift(1) / df['volume'].shift(2))
    df['vol_range_ratio'] = df['volume'] / (df['high'] - df['low'])
    
    # Signal construction
    df['geometric_eff'] = np.sign(df['intraday_eff'] * df['high_low_pos'] * df['gap_eff']) * \
                         np.power(np.abs(df['intraday_eff'] * df['high_low_pos'] * df['gap_eff']), 1/3)
    df['geometric_vol'] = np.sign(df['vol_persistence'] * df['vol_acceleration'] * df['vol_range_ratio']) * \
                         np.power(np.abs(df['vol_persistence'] * df['vol_acceleration'] * df['vol_range_ratio']), 1/3)
    df['vol_price_eff'] = df['geometric_eff'] * df['geometric_vol']
    
    # Session-Based Price Dynamics
    # Morning session signals
    df['opening_gap_strength'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['morning_range_capture'] = (df['high'] - df['open']) / (df['high'] - df['low'])
    df['morning_vol_intensity'] = df['volume'] / (df['high'] - df['low'])
    
    # Afternoon session signals
    df['closing_strength'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    df['afternoon_momentum'] = (df['close'] - df['open']) / (df['high'] - df['low'])
    df['closing_vol_ratio'] = df['volume'] / df['volume'].shift(1)
    
    # Signal construction
    df['session_alignment'] = np.sign(df['morning_range_capture'] * df['closing_strength']) * \
                             np.power(np.abs(df['morning_range_capture'] * df['closing_strength']), 1/2)
    df['volume_confirmation'] = np.sign(df['morning_vol_intensity'] * df['closing_vol_ratio']) * \
                               np.power(np.abs(df['morning_vol_intensity'] * df['closing_vol_ratio']), 1/2)
    df['session_dynamics'] = df['session_alignment'] * df['volume_confirmation'] * df['opening_gap_strength']
    
    # Multi-Timeframe Regime Detection
    # Short-term regime (1-3 days)
    df['short_price_mom'] = df['close'] / df['close'].shift(2) - 1
    df['short_vol_mom'] = df['volume'] / df['volume'].shift(2)
    df['range_efficiency'] = (df['close'] - df['close'].shift(1)) / (df['high'] - df['low'])
    
    # Medium-term regime (5-10 days)
    df['medium_price_trend'] = df['close'] / df['close'].shift(7) - 1
    df['medium_vol_trend'] = df['volume'] / df['volume'].rolling(window=7).mean()
    df['volatility_context'] = (df['high'] - df['low']) / df['high'].rolling(window=7).apply(lambda x: (x - df['low'].loc[x.index]).mean())
    
    # Signal construction
    df['geo_price_conv'] = np.sign(df['short_price_mom'] * df['medium_price_trend']) * \
                          np.power(np.abs(df['short_price_mom'] * df['medium_price_trend']), 1/2)
    df['geo_vol_conv'] = np.sign(df['short_vol_mom'] * df['medium_vol_trend']) * \
                        np.power(np.abs(df['short_vol_mom'] * df['medium_vol_trend']), 1/2)
    df['regime_detection'] = df['geo_price_conv'] * df['geo_vol_conv'] / df['volatility_context']
    
    # Final composite factor (equal-weighted combination of all components)
    df['composite_factor'] = (
        df['vol_weighted_mom'] + 
        df['vol_price_eff'] + 
        df['session_dynamics'] + 
        df['regime_detection']
    ) / 4
    
    # Handle infinite values and NaN
    df['composite_factor'] = df['composite_factor'].replace([np.inf, -np.inf], np.nan)
    
    return df['composite_factor']
