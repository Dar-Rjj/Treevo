import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Price-Volume Fractal Asymmetry with Regime-Switching Dynamics
    """
    # Price Fractal Calculation
    # Short-term (3-day) high-low range complexity
    df['hl_range_3d'] = (df['high'] - df['low']).rolling(window=3).std()
    df['price_fractal_short'] = (df['close'] - df['open']).rolling(window=3).apply(
        lambda x: np.sum(np.abs(np.diff(x))) / (np.max(x) - np.min(x)) if np.max(x) != np.min(x) else 0
    )
    
    # Medium-term (5-day) price pattern scaling
    df['price_fractal_medium'] = (df['high'] - df['low']).rolling(window=5).apply(
        lambda x: np.log(np.max(x) / np.min(x)) if np.min(x) > 0 else 0
    )
    
    # Volume Fractal Calculation
    # Short-term volume pattern irregularity
    df['volume_fractal_short'] = df['volume'].rolling(window=3).apply(
        lambda x: np.std(np.diff(x)) / (np.mean(x) + 1e-8)
    )
    
    # Medium-term volume distribution complexity
    df['volume_fractal_medium'] = df['volume'].rolling(window=5).apply(
        lambda x: np.sum((x - np.mean(x))**2) / (np.mean(x)**2 + 1e-8)
    )
    
    # Fractal Asymmetry Detection
    # Upside price expansion vs downside compression
    df['upside_expansion'] = (df['high'] - df['close'].shift(1)).rolling(window=3).mean()
    df['downside_compression'] = (df['close'].shift(1) - df['low']).rolling(window=3).mean()
    df['price_asymmetry'] = (df['upside_expansion'] - df['downside_compression']) / (df['upside_expansion'] + df['downside_compression'] + 1e-8)
    
    # Volume fractal complexity during price movements
    df['volume_complexity_up'] = df['volume_fractal_short'].rolling(window=3).corr(df['upside_expansion'])
    df['volume_complexity_down'] = df['volume_fractal_short'].rolling(window=3).corr(df['downside_compression'])
    df['volume_asymmetry'] = df['volume_complexity_up'] - df['volume_complexity_down']
    
    # Regime Classification
    # High Complexity Regime (expanding fractals)
    fractal_expansion = (df['price_fractal_medium'] > df['price_fractal_medium'].rolling(window=10).mean()) & \
                       (df['volume_fractal_medium'] > df['volume_fractal_medium'].rolling(window=10).mean())
    
    # Low Complexity Regime (compressing fractals)
    fractal_compression = (df['price_fractal_medium'] < df['price_fractal_medium'].rolling(window=10).mean()) & \
                         (df['volume_fractal_medium'] < df['volume_fractal_medium'].rolling(window=10).mean())
    
    # Transition Regime (pattern shifts)
    regime_shift = (df['price_fractal_medium'].diff(3).abs() > df['price_fractal_medium'].rolling(window=10).std()) | \
                  (df['volume_fractal_medium'].diff(3).abs() > df['volume_fractal_medium'].rolling(window=10).std())
    
    # Amount Integration
    # Large transaction patterns in fractal phases
    df['amount_fractal'] = df['amount'].rolling(window=5).apply(
        lambda x: np.std(x) / (np.mean(x) + 1e-8)
    )
    
    # Amount concentration during regime transitions
    df['amount_concentration'] = df['amount'].rolling(window=3).apply(
        lambda x: np.max(x) / (np.sum(x) + 1e-8)
    )
    
    # Multi-Scale Signal Generation
    # Short-term (3-day) fractal momentum
    df['fractal_momentum_short'] = df['price_fractal_short'].diff(2) * df['volume_fractal_short']
    
    # Medium-term (5-day) fractal persistence
    df['fractal_persistence'] = df['price_fractal_medium'].rolling(window=5).corr(df['volume_fractal_medium'])
    
    # Cross-scale fractal alignment
    df['cross_scale_alignment'] = df['price_fractal_short'].rolling(window=5).corr(df['price_fractal_medium'])
    
    # Final Alpha Construction
    # Regime-weighted fractal asymmetry score
    regime_weights = pd.Series(1.0, index=df.index)
    regime_weights[fractal_expansion] = 1.5  # Higher weight in high complexity
    regime_weights[fractal_compression] = 0.7  # Lower weight in low complexity
    regime_weights[regime_shift] = 2.0  # Highest weight during transitions
    
    fractal_asymmetry_score = (df['price_asymmetry'] * 0.6 + df['volume_asymmetry'] * 0.4) * regime_weights
    
    # Amount-enhanced pattern confirmation
    amount_confirmation = df['amount_fractal'] * df['amount_concentration'] * df['fractal_persistence']
    
    # Multi-scale fractal divergence integration
    multi_scale_divergence = (df['fractal_momentum_short'] * 0.4 + 
                             df['fractal_persistence'] * 0.3 + 
                             df['cross_scale_alignment'] * 0.3)
    
    # Final alpha factor
    alpha = (fractal_asymmetry_score * 0.5 + 
             amount_confirmation * 0.3 + 
             multi_scale_divergence * 0.2)
    
    # Clean up intermediate columns
    cols_to_drop = ['hl_range_3d', 'price_fractal_short', 'price_fractal_medium', 
                   'volume_fractal_short', 'volume_fractal_medium', 'upside_expansion',
                   'downside_compression', 'price_asymmetry', 'volume_complexity_up',
                   'volume_complexity_down', 'volume_asymmetry', 'amount_fractal',
                   'amount_concentration', 'fractal_momentum_short', 'fractal_persistence',
                   'cross_scale_alignment']
    
    for col in cols_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    
    return alpha
