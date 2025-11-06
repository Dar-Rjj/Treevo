import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Adjusted Price-Volume Divergence Factor
    """
    data = df.copy()
    
    # Calculate basic price metrics
    data['returns'] = data['close'].pct_change()
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    
    # Price Movement Quality Assessment
    # Volatility-Normalized Returns
    data['return_5d'] = data['close'].pct_change(5)
    data['atr_5d'] = data['true_range'].rolling(window=5).mean()
    data['return_efficiency_5d'] = data['return_5d'] / data['atr_5d']
    
    data['return_10d'] = data['close'].pct_change(10)
    data['atr_10d'] = data['true_range'].rolling(window=10).mean()
    data['return_efficiency_10d'] = data['return_10d'] / data['atr_10d']
    
    # Price Path Characteristics
    data['intraday_strength'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    data['price_compression'] = (data['high'] - data['low']) / (data['high'].shift(5) - data['low'].shift(5) + 1e-8)
    
    # Gap-adjusted momentum
    data['overnight_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_direction'] = np.sign(data['overnight_gap'])
    data['gap_persistence'] = data['gap_direction'].rolling(window=3).apply(
        lambda x: len([i for i in range(1, len(x)) if x.iloc[i] == x.iloc[i-1] and x.iloc[i] != 0]), 
        raw=False
    )
    
    # Volume Distribution Analysis
    data['volume_skewness'] = (data['volume'] - data['volume'].rolling(window=5).median()) / (data['volume'].rolling(window=5).std() + 1e-8)
    data['volume_clustering'] = (data['volume'] > 1.5 * data['volume'].rolling(window=20).mean()).astype(int)
    data['volume_decay_rate'] = data['volume'] / (data['volume'].rolling(window=5).max() + 1e-8)
    
    # Trade Size Dynamics
    data['amount_per_volume'] = data['amount'] / (data['volume'] + 1e-8)
    data['trade_size_trend'] = data['amount_per_volume'].rolling(window=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else np.nan, 
        raw=False
    )
    data['trade_size_consistency'] = data['amount_per_volume'].rolling(window=5).std()
    
    # Price-Volume Divergence Detection
    # Magnitude Divergence
    data['volume_rank_10d'] = data['volume'].rolling(window=10).rank(pct=True)
    data['return_rank_10d'] = abs(data['returns']).rolling(window=10).rank(pct=True)
    data['magnitude_divergence'] = data['return_rank_10d'] - data['volume_rank_10d']
    
    # Directional Divergence
    data['volume_change'] = data['volume'].pct_change()
    data['price_volume_corr_10d'] = data['returns'].rolling(window=10).corr(data['volume_change'])
    data['sign_agreement'] = (np.sign(data['returns']) == np.sign(data['volume_change'])).astype(int)
    data['sign_agreement_freq'] = data['sign_agreement'].rolling(window=10).mean()
    
    # Volatility Context Integration
    data['volatility_regime'] = data['true_range'].rolling(window=20).std() / data['true_range'].rolling(window=20).mean()
    data['volatility_expansion'] = data['true_range'] / data['true_range'].rolling(window=5).mean() - 1
    
    # Composite Signal Construction
    # Core Divergence Score
    data['correlation_component'] = -data['price_volume_corr_10d']  # Negative correlation = divergence
    data['magnitude_component'] = data['magnitude_divergence']
    data['directional_component'] = 1 - data['sign_agreement_freq']
    
    # Volatility adjustment
    data['volatility_adjustment'] = 1 / (1 + data['volatility_regime'])
    
    # Combine components with volatility adjustment
    data['core_divergence'] = (
        data['correlation_component'] * 0.4 +
        data['magnitude_component'] * 0.3 +
        data['directional_component'] * 0.3
    ) * data['volatility_adjustment']
    
    # Confirmation Filters
    data['divergence_persistence'] = (data['core_divergence'] > 0).rolling(window=3).sum()
    data['volume_pattern_consistency'] = data['volume_skewness'].rolling(window=5).std()
    
    # Final signal with filters
    data['final_signal'] = data['core_divergence'] * np.exp(-data['volume_pattern_consistency'])
    data['final_signal'] = data['final_signal'] * (data['divergence_persistence'] >= 2)
    
    # Time-decay weighting
    weights = np.array([0.5, 0.3, 0.2])  # Recent days more important
    data['weighted_signal'] = data['final_signal'].rolling(window=3).apply(
        lambda x: np.dot(x, weights) if len(x) == 3 else np.nan, 
        raw=False
    )
    
    # Signal refinement
    signal_std = data['weighted_signal'].rolling(window=20).std()
    data['refined_factor'] = data['weighted_signal'] / (signal_std + 1e-8)
    
    return data['refined_factor']
