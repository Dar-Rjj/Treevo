import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Divergence Spectrum with Microstructure Anchoring alpha factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Price and volume calculations
    df['price_change'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    df['amount_change'] = df['amount'].pct_change()
    df['range_utilization'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)
    
    # Multi-Dimensional Divergence Metrics
    # Price-Volume Velocity Divergence
    df['price_velocity'] = df['price_change'].rolling(window=5).mean()
    df['volume_velocity'] = df['volume_change'].rolling(window=5).mean()
    df['pv_divergence'] = (df['price_velocity'] - df['volume_velocity']) * np.sign(df['price_velocity'])
    
    # Amount-Price Efficiency Divergence
    df['amount_per_volume'] = df['amount'] / df['volume'].replace(0, np.nan)
    df['efficiency_divergence'] = (df['price_change'] - df['amount_per_volume'].pct_change()) * df['price_change'].abs()
    
    # Intraday Range-Volume Divergence
    df['range_divergence'] = (df['range_utilization'] - df['volume_change']) * df['range_utilization']
    
    # Microstructure Anchoring Framework
    # Price clustering zones (support/resistance levels)
    df['price_cluster_5'] = (df['close'] / 0.05).round() * 0.05
    df['price_cluster_10'] = (df['close'] / 0.10).round() * 0.10
    
    # Volume-weighted anchor strength
    df['anchor_strength_5'] = df.groupby('price_cluster_5')['volume'].transform(
        lambda x: x.rolling(window=10, min_periods=1).sum()
    )
    df['anchor_strength_10'] = df.groupby('price_cluster_10')['volume'].transform(
        lambda x: x.rolling(window=10, min_periods=1).sum()
    )
    
    # Anchor proximity
    df['proximity_5'] = 1 - (abs(df['close'] - df['price_cluster_5']) / df['close'])
    df['proximity_10'] = 1 - (abs(df['close'] - df['price_cluster_10']) / df['close'])
    
    # Anchor convergence/divergence
    df['anchor_divergence_5'] = (df['close'] - df['price_cluster_5']) / df['close']
    df['anchor_divergence_10'] = (df['close'] - df['price_cluster_10']) / df['close']
    
    # Divergence Spectrum Analysis
    # Multi-timeframe divergence alignment
    df['pv_divergence_short'] = df['pv_divergence'].rolling(window=3).mean()
    df['pv_divergence_medium'] = df['pv_divergence'].rolling(window=8).mean()
    
    df['efficiency_divergence_short'] = df['efficiency_divergence'].rolling(window=3).mean()
    df['efficiency_divergence_medium'] = df['efficiency_divergence'].rolling(window=8).mean()
    
    df['range_divergence_short'] = df['range_divergence'].rolling(window=3).mean()
    df['range_divergence_medium'] = df['range_divergence'].rolling(window=8).mean()
    
    # Divergence type classification and weighting
    df['directional_alignment'] = (
        np.sign(df['pv_divergence_short']) * np.sign(df['efficiency_divergence_short']) * 
        np.sign(df['range_divergence_short'])
    )
    
    # Multi-timeframe alignment score
    df['timeframe_alignment'] = (
        (df['pv_divergence_short'] * df['pv_divergence_medium'] > 0).astype(int) +
        (df['efficiency_divergence_short'] * df['efficiency_divergence_medium'] > 0).astype(int) +
        (df['range_divergence_short'] * df['range_divergence_medium'] > 0).astype(int)
    ) / 3.0
    
    # Anchored Divergence Alpha Factor
    # Weighted divergence spectrum combination
    divergence_weights = {
        'pv': 0.4,
        'efficiency': 0.35,
        'range': 0.25
    }
    
    df['weighted_divergence'] = (
        divergence_weights['pv'] * df['pv_divergence_short'] +
        divergence_weights['efficiency'] * df['efficiency_divergence_short'] +
        divergence_weights['range'] * df['range_divergence_short']
    )
    
    # Microstructure anchor multiplier
    df['anchor_multiplier'] = (
        (df['proximity_5'] * df['anchor_strength_5'].rank(pct=True) +
         df['proximity_10'] * df['anchor_strength_10'].rank(pct=True)) / 2
    )
    
    # Anchor-confirmed vs rejected signals
    df['anchor_confirmation'] = (
        np.sign(df['weighted_divergence']) * np.sign(df['anchor_divergence_5']) * 
        np.sign(df['anchor_divergence_10'])
    )
    
    # Final alpha signal
    df['alpha_signal'] = (
        df['weighted_divergence'] * 
        (1 + df['anchor_multiplier']) * 
        (1 + df['timeframe_alignment']) * 
        (1 + 0.5 * df['anchor_confirmation'])
    )
    
    # Normalize and smooth the final signal
    alpha_factor = df['alpha_signal'].rolling(window=5, min_periods=1).mean()
    alpha_factor = (alpha_factor - alpha_factor.rolling(window=20, min_periods=1).mean()) / alpha_factor.rolling(window=20, min_periods=1).std()
    
    return alpha_factor
