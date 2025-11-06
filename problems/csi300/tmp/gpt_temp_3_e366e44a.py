import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price-Volume Efficiency Ratio
    # True Range Component
    df['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    
    # Closing Position Component
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)
    
    # Price Movement Efficiency
    df['price_efficiency'] = (df['close'] - df['open']) / df['true_range'].replace(0, np.nan)
    
    # Volume Concentration
    df['volume_ma5'] = df['volume'].rolling(window=5, min_periods=1).mean()
    df['volume_concentration'] = df['volume'] / df['volume_ma5']
    
    # Volume Persistence
    df['volume_trend'] = df['volume'].rolling(window=3, min_periods=1).apply(
        lambda x: 1 if (x.diff().iloc[1:] > 0).all() else (-1 if (x.diff().iloc[1:] < 0).all() else 0)
    )
    
    # Volume Efficiency
    df['volume_efficiency'] = df['volume_concentration'] * df['volume_trend']
    
    # Price-Volume Efficiency Ratio
    df['pv_efficiency_ratio'] = df['price_efficiency'] * df['volume_efficiency']
    
    # Opening Gap Momentum Persistence
    df['gap_magnitude'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['gap_filled'] = np.where(
        df['gap_magnitude'] > 0,
        (df['low'] <= df['close'].shift(1)).astype(int),
        (df['high'] >= df['close'].shift(1)).astype(int)
    )
    df['gap_preservation'] = abs(df['gap_magnitude']) * (1 - df['gap_filled'])
    df['volume_gap_interaction'] = df['volume_concentration'] * df['gap_preservation']
    df['gap_momentum'] = df['gap_preservation'] * df['volume_gap_interaction']
    
    # Extreme Price Rejection Factor
    df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['true_range'].replace(0, np.nan)
    df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['true_range'].replace(0, np.nan)
    df['shadow_dominance'] = df['upper_shadow'] - df['lower_shadow']
    
    # Volume Intensity for Rejections
    df['high_volume_rejection'] = np.where(
        (df['shadow_dominance'].abs() > 0.3) & (df['volume_concentration'] > 1.5),
        df['shadow_dominance'] * df['volume_concentration'], 0
    )
    df['low_volume_rejection'] = np.where(
        (df['shadow_dominance'].abs() > 0.3) & (df['volume_concentration'] < 0.8),
        df['shadow_dominance'] * (1 - df['volume_concentration']), 0
    )
    df['rejection_factor'] = df['high_volume_rejection'] + df['low_volume_rejection']
    
    # Volume-Cluster Breakout Detection
    df['volume_spike'] = (df['volume'] > df['volume'].rolling(window=10, min_periods=1).quantile(0.8)).astype(int)
    df['volume_increase'] = (df['volume'] > df['volume'].shift(1) * 1.2).astype(int)
    df['volume_cluster'] = df['volume_spike'] | df['volume_increase']
    
    # Price Breakouts
    df['high_5d'] = df['high'].rolling(window=5, min_periods=1).max()
    df['low_5d'] = df['low'].rolling(window=5, min_periods=1).min()
    df['breakout_up'] = ((df['close'] > df['high_5d'].shift(1)) & df['volume_cluster']).astype(int)
    df['breakout_down'] = ((df['close'] < df['low_5d'].shift(1)) & df['volume_cluster']).astype(int)
    df['breakout_signal'] = df['breakout_up'] - df['breakout_down']
    
    # Price-Volume Convergence Divergence
    df['price_change'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    
    # Short-term Convergence
    df['price_trend_3'] = df['price_change'].rolling(window=3, min_periods=1).mean()
    df['volume_trend_3'] = df['volume_change'].rolling(window=3, min_periods=1).mean()
    df['convergence'] = np.sign(df['price_trend_3']) * np.sign(df['volume_trend_3'])
    
    # Divergence Emergence
    df['price_volume_divergence'] = np.where(
        df['convergence'] < 0,
        abs(df['price_trend_3'] - df['volume_trend_3']), 0
    )
    df['divergence_strength'] = df['price_volume_divergence'] * df['volume_concentration']
    
    # Composite Alpha Factor
    alpha = (
        df['pv_efficiency_ratio'] * 0.25 +
        df['gap_momentum'] * 0.20 +
        df['rejection_factor'] * 0.25 +
        df['breakout_signal'] * 0.15 +
        df['divergence_strength'] * 0.15
    )
    
    return alpha
