import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Efficiency-Weighted Flow Dynamics factor combining multi-scale flow momentum,
    volume-price efficiency, flow-pressure dynamics, path-flow complexity, and dynamic regime synthesis.
    """
    # Calculate basic flow components
    df['net_flow'] = df['amount'] / df['volume']  # Price-weighted flow
    df['buying_flow'] = np.where(df['close'] > df['open'], df['amount'], 0)
    df['selling_flow'] = np.where(df['close'] <= df['open'], df['amount'], 0)
    
    # Multi-Scale Flow Momentum
    # Short-term flow momentum (5-day)
    df['flow_momentum_5'] = df['net_flow'].rolling(window=5).mean() / df['net_flow'].rolling(window=5).std()
    
    # Medium-term flow momentum (20-day)
    df['flow_momentum_20'] = df['net_flow'].rolling(window=20).mean() / df['net_flow'].rolling(window=20).std()
    
    # Flow momentum divergence
    df['momentum_divergence'] = df['flow_momentum_5'] - df['flow_momentum_20']
    
    # Volume-Price Efficiency Framework
    # Intraday flow efficiency
    df['intraday_flow_efficiency'] = (df['buying_flow'] - df['selling_flow']) / (df['high'] - df['low']).replace(0, np.nan)
    
    # Movement efficiency
    df['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['movement_efficiency'] = abs(df['net_flow'] - df['net_flow'].shift(1)) / df['true_range'].replace(0, np.nan)
    
    # Volume-flow correlation (3-day)
    df['flow_price_corr'] = df['net_flow'].rolling(window=3).corr(df['close'])
    
    # Flow-Pressure Dynamics
    # Pressure accumulation from directional flow strength
    df['directional_flow_strength'] = (df['net_flow'] - df['net_flow'].rolling(window=10).mean()) / df['net_flow'].rolling(window=10).std()
    df['pressure_accumulation'] = df['directional_flow_strength'].rolling(window=5).sum()
    
    # Flow efficiency ratios
    df['flow_efficiency_ratio'] = (df['close'] - df['open']) / (df['net_flow'].replace(0, np.nan))
    
    # Flow-volume asymmetry
    df['flow_volume_asymmetry'] = (df['net_flow'].rolling(window=5).std() - df['volume'].rolling(window=5).std()) / (df['net_flow'].rolling(window=5).std() + df['volume'].rolling(window=5).std())
    
    # Path-Flow Complexity Analysis
    # Flow path efficiency
    df['price_path'] = (df['high'] - df['low']) / df['close'].shift(1)
    df['flow_path_efficiency'] = df['net_flow'] / df['price_path'].replace(0, np.nan)
    
    # Flow oscillation intensity
    df['flow_oscillation'] = df['net_flow'].rolling(window=5).apply(lambda x: np.std(x.diff().dropna()) if len(x.diff().dropna()) > 0 else np.nan)
    
    # Multi-scale flow path alignment
    df['flow_alignment_5'] = df['net_flow'].rolling(window=5).corr(df['close'])
    df['flow_alignment_20'] = df['net_flow'].rolling(window=20).corr(df['close'])
    df['multi_scale_alignment'] = df['flow_alignment_5'] - df['flow_alignment_20']
    
    # Dynamic Regime Synthesis
    # Combine flow momentum with volume efficiency confirmation
    df['momentum_efficiency_combo'] = df['flow_momentum_5'] * df['movement_efficiency']
    
    # Weight signals by flow-pressure alignment strength
    pressure_alignment = df['pressure_accumulation'] * df['directional_flow_strength']
    df['weighted_signals'] = (df['momentum_efficiency_combo'] * pressure_alignment) / (abs(pressure_alignment) + 1e-6)
    
    # Generate regime-specific factors from flow complexity shifts
    df['flow_complexity'] = (df['flow_oscillation'] * df['momentum_divergence'] * df['multi_scale_alignment'])
    df['regime_factor'] = df['weighted_signals'] * (1 + df['flow_complexity'])
    
    # Final factor: Efficiency-Weighted Flow Dynamics
    factor = (
        df['regime_factor'] * 
        df['flow_efficiency_ratio'].replace([np.inf, -np.inf], np.nan) * 
        (1 + df['flow_volume_asymmetry'])
    )
    
    # Clean and return
    factor = factor.replace([np.inf, -np.inf], np.nan)
    return factor
