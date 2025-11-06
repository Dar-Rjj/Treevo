import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Multi-Horizon Efficiency Dynamics
    # Intraday Efficiency Patterns
    df['intraday_efficiency'] = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    df['efficiency_momentum_3d'] = df['intraday_efficiency'].rolling(window=3).mean()
    df['efficiency_persistence_5d'] = df['intraday_efficiency'].rolling(window=5).apply(lambda x: (x > 0).sum() / len(x))
    
    # Efficiency Clustering Regimes
    df['efficiency_median_10d'] = df['intraday_efficiency'].rolling(window=10).median()
    df['high_efficiency_cluster'] = (df['intraday_efficiency'] > df['efficiency_median_10d']).astype(int)
    df['low_efficiency_cluster'] = (df['intraday_efficiency'] < df['efficiency_median_10d']).astype(int)
    df['cluster_strength'] = df['high_efficiency_cluster'].rolling(window=3).sum()
    
    # Multi-Timeframe Pressure Divergence
    # Short-Term Pressure (5-day)
    df['buying_pressure'] = ((2 * df['close'] - df['low'] - df['high']) / (df['high'] - df['low']).replace(0, np.nan)) * df['volume']
    df['selling_pressure'] = ((df['high'] + df['low'] - 2 * df['close']) / (df['high'] - df['low']).replace(0, np.nan)) * df['volume']
    df['pressure_momentum_5d'] = (df['buying_pressure'] - df['selling_pressure']).rolling(window=5).mean()
    
    # Medium-Term Pressure (20-day)
    df['price_pressure_20d'] = df['close'] / df['close'].shift(20) - 1
    df['intraday_pressure_persistence'] = ((df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)).rolling(window=20).sum()
    df['pressure_divergence'] = df['pressure_momentum_5d'] - df['price_pressure_20d']
    
    # Cluster-Adaptive Signals
    # Combine Efficiency and Pressure Regimes
    df['momentum_enhancement'] = (df['high_efficiency_cluster'] == 1) & (df['pressure_divergence'].abs() > df['pressure_divergence'].rolling(window=10).median())
    df['reversal_emphasis'] = (df['low_efficiency_cluster'] == 1) & (df['pressure_divergence'].abs() < df['pressure_divergence'].rolling(window=10).median())
    df['mixed_clustering'] = ~df['momentum_enhancement'] & ~df['reversal_emphasis']
    
    df['cluster_signal'] = 0
    df.loc[df['momentum_enhancement'], 'cluster_signal'] = df['pressure_divergence'] * 1.5
    df.loc[df['reversal_emphasis'], 'cluster_signal'] = -df['pressure_divergence'] * 1.2
    df.loc[df['mixed_clustering'], 'cluster_signal'] = df['pressure_divergence'] * 0.8
    
    df['regime_alignment_strength'] = np.sign(df['cluster_signal']).rolling(window=3).apply(lambda x: (x == x.iloc[0]).sum() if len(x) == 3 else np.nan)
    
    # Microstructure Confirmation
    df['range_compression'] = (df['high'] - df['low']) / (df['high'] - df['low']).rolling(window=5).mean()
    df['volume_intensity'] = df['volume'] / df['volume'].rolling(window=20).median()
    df['cluster_signal_micro'] = df['cluster_signal'] * df['range_compression'] * df['volume_intensity']
    
    # Liquidity and Volatility Adjustment
    # Liquidity-Based Scaling
    df['amount_intensity'] = df['amount'] / df['amount'].rolling(window=10).median()
    df['volume_liquidity_ratio'] = df['volume'] / df['amount'].replace(0, np.nan)
    
    # Volatility Clustering Adjustment
    df['current_volatility'] = df['high'] - df['low']
    df['volatility_persistence'] = df['current_volatility'].rolling(window=3).apply(lambda x: x.autocorr() if len(x) == 3 else np.nan)
    df['volatility_clustering_strength'] = df['volatility_persistence'].abs() + 0.1
    
    # Composite Signal Generation
    df['adjusted_signal'] = df['cluster_signal_micro'] * df['amount_intensity'] / df['volume_liquidity_ratio'].replace(0, np.nan)
    df['volatility_adjusted_signal'] = df['adjusted_signal'] / df['volatility_clustering_strength']
    
    # Temporal Enhancement
    weights = np.array([0.5, 0.3, 0.2])  # Exponential weights for 3 days
    df['temporal_signal'] = df['volatility_adjusted_signal'].rolling(window=3).apply(lambda x: np.dot(x, weights) if len(x) == 3 else np.nan)
    df['persistence_filter'] = np.sign(df['temporal_signal']).rolling(window=3).apply(lambda x: (x == x.iloc[0]).sum() if len(x) == 3 else np.nan)
    
    # Final Alpha Signal
    df['final_signal'] = df['temporal_signal'] * df['persistence_filter']
    
    return df['final_signal']
