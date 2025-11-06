import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Entropic Momentum with Structural Break Detection
    """
    data = df.copy()
    
    # Multi-Horizon Entropy Measurement
    # Short-term price path entropy (3-day)
    data['price_range_ratio'] = (data['high'] - data['low']) / data['close']
    data['price_path_entropy_3d'] = data['price_range_ratio'].rolling(window=3).std()
    
    # Medium-term volatility entropy (5-day)
    data['returns'] = data['close'].pct_change()
    data['volatility_entropy_5d'] = data['returns'].rolling(window=5).std()
    
    # Volume-entropy coupling patterns
    data['volume_entropy'] = data['volume'].rolling(window=5).apply(
        lambda x: np.std(x) / (np.mean(x) + 1e-8)
    )
    data['amount_entropy'] = data['amount'].rolling(window=5).apply(
        lambda x: np.std(x) / (np.mean(x) + 1e-8)
    )
    
    # Structural Break Identification
    # Entropy regime shift detection
    data['vol_entropy_ma'] = data['volatility_entropy_5d'].rolling(window=10).mean()
    data['vol_entropy_std'] = data['volatility_entropy_5d'].rolling(window=10).std()
    data['structural_break'] = (
        (data['volatility_entropy_5d'] - data['vol_entropy_ma']) / 
        (data['vol_entropy_std'] + 1e-8)
    ).abs()
    
    # Break magnitude quantification
    data['break_magnitude'] = data['structural_break'].rolling(window=5).max()
    
    # Multi-scale break synchronization
    data['volume_break'] = (
        (data['volume_entropy'] - data['volume_entropy'].rolling(window=10).mean()) /
        (data['volume_entropy'].rolling(window=10).std() + 1e-8)
    ).abs()
    
    # Amount-Weighted Entropic Momentum
    # Price-amount entropy correlation
    data['price_amount_corr'] = data['close'].rolling(window=5).corr(data['amount'])
    
    # Volume-entropy efficiency ratios
    data['volume_efficiency'] = (
        data['volume_entropy'] / (data['volatility_entropy_5d'] + 1e-8)
    )
    
    # Entropic persistence strength
    data['entropy_persistence'] = (
        data['volatility_entropy_5d'].rolling(window=3).apply(
            lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 else 0
        )
    )
    
    # Structural Break Momentum
    # Pre-break entropy accumulation
    data['pre_break_entropy'] = data['volatility_entropy_5d'].shift(1).rolling(window=3).mean()
    
    # Post-break momentum divergence
    data['post_break_momentum'] = (
        data['returns'].rolling(window=3).mean() * 
        (1 + data['structural_break'].shift(1))
    )
    
    # Multi-Scale Entropic Divergence
    # Entropic momentum efficiency
    data['entropic_momentum'] = (
        data['volatility_entropy_5d'] * data['returns'].rolling(window=3).std()
    )
    
    # Entropic regime classification
    data['entropy_regime'] = pd.cut(
        data['volatility_entropy_5d'], 
        bins=[-np.inf, data['volatility_entropy_5d'].quantile(0.33), 
              data['volatility_entropy_5d'].quantile(0.66), np.inf],
        labels=[0, 1, 2]
    ).astype(float)
    
    # Break-enhanced divergence patterns
    data['break_divergence'] = (
        data['structural_break'] * data['entropic_momentum'] * 
        np.sign(data['returns'].rolling(window=3).mean())
    )
    
    # Intraday Entropic Structure
    # Open-Close Entropic Dynamics
    data['intraday_range'] = (data['close'] - data['open']) / data['open']
    data['intraday_entropy'] = data['intraday_range'].abs().rolling(window=5).std()
    
    # Intraday Entropy-Volume Interactions
    data['intraday_volume_entropy'] = (
        data['intraday_entropy'] * data['volume_entropy']
    )
    
    # Amount-Driven Entropic Shifts
    data['amount_entropic_shift'] = (
        data['amount_entropy'].pct_change() * data['structural_break']
    )
    
    # Entropic Alpha Signal Synthesis
    # Structural Break Momentum Signals
    data['pre_break_signal'] = (
        data['pre_break_entropy'] * data['entropy_persistence']
    )
    
    data['break_acceleration'] = (
        data['structural_break'] * data['post_break_momentum']
    )
    
    # Entropic Regime Scoring
    regime_weights = {
        0: 0.3,  # Low entropy regime
        1: 0.6,  # Medium entropy regime  
        2: 0.9   # High entropy regime
    }
    data['regime_score'] = data['entropy_regime'].map(regime_weights)
    
    # Multi-Scale Entropic Integration
    # Final alpha factor
    alpha = (
        0.25 * data['break_divergence'] +
        0.20 * data['pre_break_signal'] +
        0.15 * data['break_acceleration'] +
        0.15 * data['intraday_volume_entropy'] +
        0.10 * data['amount_entropic_shift'] +
        0.10 * data['volume_efficiency'] +
        0.05 * data['regime_score']
    ) * (1 + 0.1 * data['price_amount_corr'])
    
    return alpha
