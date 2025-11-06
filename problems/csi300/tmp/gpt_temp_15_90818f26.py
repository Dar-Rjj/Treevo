import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Calculate basic price changes and ratios
    df['close_pct_change'] = df['close'] / df['close'].shift(1) - 1
    df['close_pct_change_prev'] = df['close'].shift(1) / df['close'].shift(2) - 1
    df['volume_ratio'] = df['volume'] / df['volume'].shift(1)
    df['high_low_range'] = df['high'] - df['low']
    df['high_low_range_prev'] = df['high_low_range'].shift(1)
    df['high_low_range_prev2'] = df['high_low_range'].shift(2)
    df['range_ratio'] = df['high_low_range'] / (df['high_low_range_prev'] + 1e-8)
    
    # Calculate entropy measures using rolling window
    def calculate_entropy(series, window=10):
        probabilities = series.rolling(window).apply(
            lambda x: (x.value_counts(normalize=True) ** 2).sum(), 
            raw=False
        )
        return 1 - probabilities
    
    # Discretize price and volume for entropy calculation
    df['price_bin'] = pd.cut(df['close'], bins=20, labels=False)
    df['volume_bin'] = pd.cut(df['volume'], bins=20, labels=False)
    
    df['price_entropy'] = calculate_entropy(df['price_bin'])
    df['volume_entropy'] = calculate_entropy(df['volume_bin'])
    
    # Entropy-Volatility Momentum Structure
    df['entropy_weighted_momentum_acceleration'] = (
        (df['close'] - df['close'].shift(2)) / 
        (df['high_low_range_prev2'] + 1e-8) * 
        df['price_entropy'] * 
        df['range_ratio']
    )
    
    # Volatility-Entropy Persistence
    df['close_gt_prev1'] = (df['close'] > df['close'].shift(1)).astype(int)
    df['close_gt_prev2'] = (df['close'] > df['close'].shift(2)).astype(int)
    df['close_gt_prev3'] = (df['close'] > df['close'].shift(3)).astype(int)
    df['volatility_entropy_persistence'] = (
        (df['close_gt_prev1'] + df['close_gt_prev2'] + df['close_gt_prev3']) / 3 * 
        df['range_ratio'] * 
        df['price_entropy']
    )
    
    # Entropy-Elastic Price Momentum
    df['entropy_elastic_price_momentum'] = (
        (df['close_pct_change'] - df['close_pct_change_prev']) * 
        df['range_ratio'] * 
        df['price_entropy']
    )
    
    # Entropy-Flow Asymmetry
    df['entropy_adaptive_intraday_flow'] = (
        (df['close'] - df['open']) / 
        (df['high_low_range'] + 1e-8) * 
        (df['volume_ratio'] - 1) * 
        df['price_entropy']
    )
    
    # Volume-Entropy Collapse
    df['upper_shadow'] = (df['high'] - df['close']) / (df['high_low_range'] + 1e-8)
    df['lower_shadow'] = (df['close'] - df['low']) / (df['high_low_range'] + 1e-8)
    df['volume_entropy_collapse'] = (
        (df['upper_shadow'] - df['lower_shadow']) * 
        (df['volume_ratio'] - 1) * 
        df['volume_entropy']
    )
    df['volume_entropy_collapse'] = df['volume_entropy_collapse'].where(
        df['high_low_range'] > df['high_low_range_prev'], 0
    )
    
    # Entropy-Flow Divergence
    df['entropy_flow_divergence'] = (
        df['entropy_adaptive_intraday_flow'] - df['volume_entropy_collapse']
    )
    
    # Entropy-Volume Regime Dynamics
    df['volume_ratio_prev'] = df['volume'].shift(1) / df['volume'].shift(2)
    df['volatility_entropy_volume_momentum'] = (
        ((df['volume_ratio'] - 1) - (df['volume_ratio_prev'] - 1)) * 
        df['range_ratio'] * 
        df['volume_entropy']
    )
    
    # Entropy Volume-Price Divergence
    df['entropy_volume_price_divergence'] = (
        df['volatility_entropy_volume_momentum'] - 
        df['entropy_elastic_price_momentum'] * np.sign(df['entropy_elastic_price_momentum'])
    )
    
    # Entropy-Volume Convergence
    df['entropy_volume_convergence'] = (
        df['volatility_entropy_volume_momentum'] * df['entropy_elastic_price_momentum']
    )
    
    # Entropy-Gap Dynamics
    df['entropy_gap_absorption'] = (
        (df['close'] - df['open']) / 
        (df['open'] - df['close'].shift(1) + 1e-8) * 
        df['range_ratio'] * 
        df['price_entropy']
    )
    
    # Entropy Range Momentum
    df['range_ratio_prev'] = df['high_low_range_prev'] / (df['high_low_range_prev2'] + 1e-8)
    df['entropy_range_momentum'] = (
        (df['range_ratio'] - df['range_ratio_prev']) * 
        (df['volume_ratio'] - 1) * 
        df['volume_entropy']
    )
    
    # Entropy-Gap Alignment
    df['entropy_gap_alignment'] = (
        df['entropy_gap_absorption'] * df['entropy_range_momentum']
    )
    
    # Regime-Adaptive Entropy Synthesis
    df['high_entropy_volatility'] = (
        df['entropy_weighted_momentum_acceleration'] * 
        df['entropy_flow_divergence'] * 
        df['entropy_volume_convergence']
    )
    
    df['low_entropy_volatility'] = (
        df['entropy_elastic_price_momentum'] * 
        df['entropy_volume_price_divergence'] * 
        df['entropy_gap_alignment']
    )
    
    df['transition_regime'] = (
        df['volatility_entropy_persistence'] * 
        df['entropy_adaptive_intraday_flow'] * 
        df['entropy_range_momentum']
    )
    
    # Adaptive regime selection based on entropy levels
    df['regime_adaptive_entropy_synthesis'] = np.select(
        [
            df['price_entropy'] > df['price_entropy'].rolling(20).mean(),
            df['price_entropy'] < df['price_entropy'].rolling(20).mean() - df['price_entropy'].rolling(20).std(),
        ],
        [
            df['high_entropy_volatility'],
            df['low_entropy_volatility'],
        ],
        default=df['transition_regime']
    )
    
    # Adaptive Alpha Synthesis
    df['momentum_efficiency_component'] = (
        df['entropy_elastic_price_momentum'] * df['entropy_flow_divergence']
    )
    
    df['volume_regime_component'] = (
        df['volatility_entropy_volume_momentum'] * df['entropy_volume_price_divergence']
    )
    
    # Final Alpha
    df['alpha'] = (
        df['regime_adaptive_entropy_synthesis'] * 
        df['momentum_efficiency_component'] * 
        df['volume_regime_component']
    )
    
    # Clean up intermediate columns
    result = df['alpha'].copy()
    
    return result
