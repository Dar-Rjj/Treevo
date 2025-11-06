import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate net flow (amount represents money flow)
    df = df.copy()
    df['net_flow'] = df['amount']
    
    # Multi-Scale Flow Momentum
    # Short-term flow persistence (5-day)
    df['flow_momentum_5d'] = df['net_flow'].rolling(window=5).mean()
    
    # Medium-term flow alignment (20-day)
    df['flow_momentum_20d'] = df['net_flow'].rolling(window=20).mean()
    
    # Momentum divergence across horizons
    df['momentum_divergence'] = df['flow_momentum_5d'] - df['flow_momentum_20d']
    
    # Volume-Price Efficiency
    # Intraday flow efficiency: (Buying flow - Selling flow)/(High - Low)
    df['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['intraday_flow_efficiency'] = df['net_flow'] / df['true_range'].replace(0, np.nan)
    
    # Movement efficiency: |ΔNet flow|/TrueRange
    df['delta_net_flow'] = df['net_flow'].diff().abs()
    df['movement_efficiency'] = df['delta_net_flow'] / df['true_range'].replace(0, np.nan)
    
    # Flow-price relationship (3-day correlation)
    df['price_return'] = df['close'].pct_change()
    df['flow_price_corr_3d'] = df['net_flow'].rolling(window=3).corr(df['price_return'])
    
    # Pressure Accumulation & Overflow
    # Directional flow strength accumulation
    df['directional_flow'] = np.sign(df['net_flow']) * df['net_flow'].abs()
    df['flow_strength_accum'] = df['directional_flow'].rolling(window=10).sum()
    
    # Flow efficiency ratios (price movement per unit flow)
    df['price_movement'] = df['close'] - df['open']
    df['flow_efficiency_ratio'] = df['price_movement'] / df['net_flow'].replace(0, np.nan)
    
    # Flow-volume asymmetry detection
    df['flow_volume_ratio'] = df['net_flow'] / df['volume'].replace(0, np.nan)
    df['flow_volume_asymmetry'] = df['flow_volume_ratio'].rolling(window=5).std()
    
    # Flow Path Analysis
    # Flow path efficiency (net flow vs price path)
    df['price_path'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    df['flow_path_5d'] = df['net_flow'].rolling(window=5).sum()
    df['flow_path_efficiency'] = df['price_path'] / df['flow_path_5d'].replace(0, np.nan)
    
    # Flow oscillation intensity
    df['flow_oscillation'] = df['net_flow'].rolling(window=5).std() / df['net_flow'].rolling(window=5).mean().abs().replace(0, np.nan)
    
    # Multi-scale flow alignment consistency
    df['flow_alignment_5d'] = df['net_flow'].rolling(window=5).apply(lambda x: np.corrcoef(range(5), x)[0,1] if len(x) == 5 and not np.isnan(x).any() else np.nan)
    df['flow_alignment_20d'] = df['net_flow'].rolling(window=20).apply(lambda x: np.corrcoef(range(20), x)[0,1] if len(x) == 20 and not np.isnan(x).any() else np.nan)
    df['alignment_consistency'] = df['flow_alignment_5d'] * df['flow_alignment_20d']
    
    # Combine components with appropriate weights
    factor = (
        0.15 * df['momentum_divergence'] +
        0.12 * df['intraday_flow_efficiency'] +
        0.10 * df['movement_efficiency'] +
        0.12 * df['flow_price_corr_3d'] +
        0.10 * df['flow_strength_accum'] +
        0.08 * df['flow_efficiency_ratio'] +
        0.08 * df['flow_volume_asymmetry'] +
        0.12 * df['flow_path_efficiency'] +
        0.08 * df['flow_oscillation'] +
        0.05 * df['alignment_consistency']
    )
    
    return factor
