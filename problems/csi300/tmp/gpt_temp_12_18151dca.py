import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Fractal Gap-Liquidity Divergence with Volatility-Regime Adaptation
    """
    data = df.copy()
    
    # Multi-Scale Gap Divergence Analysis
    # Micro Gap Divergence (1-day)
    data['overnight_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['intraday_gap'] = (data['close'] - data['open']) / data['open']
    data['micro_divergence'] = (np.sign(data['overnight_gap']) != np.sign(data['intraday_gap'])).astype(int)
    
    # Meso Gap Divergence (3-day)
    data['overnight_gap_3d'] = data['overnight_gap'].rolling(window=3, min_periods=1).mean()
    data['intraday_gap_3d'] = data['intraday_gap'].rolling(window=3, min_periods=1).mean()
    data['meso_divergence_strength'] = (
        (np.sign(data['overnight_gap'].rolling(window=3, min_periods=1).apply(lambda x: len(set(np.sign(x.dropna()))), raw=False)) == 1) &
        (np.sign(data['intraday_gap'].rolling(window=3, min_periods=1).apply(lambda x: len(set(np.sign(x.dropna()))), raw=False)) == 1)
    ).astype(int)
    
    # Macro Gap Divergence (5-day)
    data['gap_divergence_persistence'] = (
        data['micro_divergence'].rolling(window=5, min_periods=1).sum() / 5
    )
    
    # Multi-Fractal Liquidity Acceleration
    # Volume Fractal Acceleration Patterns
    data['micro_vol_accel'] = (data['volume'] - data['volume'].shift(1)) / data['volume'].shift(1).replace(0, np.nan)
    data['meso_vol_accel'] = (data['volume'] - data['volume'].shift(3)) / data['volume'].shift(3).replace(0, np.nan)
    data['macro_vol_accel'] = (data['volume'] - data['volume'].shift(5)) / data['volume'].shift(5).replace(0, np.nan)
    
    # Amount-Volume Fractal Dynamics
    data['transaction_size'] = data['amount'] / data['volume'].replace(0, np.nan)
    data['transaction_momentum'] = (data['transaction_size'] / data['transaction_size'].shift(1)) - 1
    
    # Multi-scale Liquidity Divergence
    data['liquidity_divergence'] = (
        data['micro_vol_accel'] - data['meso_vol_accel'].rolling(window=3, min_periods=1).mean()
    )
    
    # Gap-Liquidity Divergence Integration
    # Cross-Fractal Divergence Alignment
    data['gap_vol_alignment'] = (
        np.sign(data['overnight_gap']) * np.sign(data['micro_vol_accel'])
    )
    
    # Divergence Strength Scoring
    data['micro_div_score'] = np.abs(data['overnight_gap']) * data['micro_vol_accel']
    data['meso_div_score'] = np.abs(data['overnight_gap_3d']) * data['meso_vol_accel']
    data['macro_div_score'] = data['gap_divergence_persistence'] * data['macro_vol_accel']
    
    # Volatility-Regime Adaptive Framework
    # Multi-Scale Volatility Context
    data['micro_vol'] = (data['high'] - data['low']) / data['close']
    data['meso_vol'] = data['micro_vol'].rolling(window=3, min_periods=1).mean()
    data['macro_vol'] = data['micro_vol'].rolling(window=5, min_periods=1).mean()
    
    # Volatility regime classification
    vol_quantiles = data['meso_vol'].quantile([0.33, 0.67])
    data['vol_regime'] = pd.cut(
        data['meso_vol'], 
        bins=[-np.inf, vol_quantiles[0.33], vol_quantiles[0.67], np.inf],
        labels=['low', 'medium', 'high']
    )
    
    # Regime-Adaptive Factor Weighting
    regime_weights = {
        'low': [0.4, 0.4, 0.2],      # Emphasize micro and meso
        'medium': [0.3, 0.4, 0.3],   # Balanced approach
        'high': [0.2, 0.3, 0.5]      # Emphasize macro and persistence
    }
    
    # Dynamic Alpha Generation
    # Apply Volatility-Regime Weighting
    def apply_regime_weighting(row):
        weights = regime_weights.get(row['vol_regime'], [0.33, 0.33, 0.34])
        return (
            weights[0] * row['micro_div_score'] +
            weights[1] * row['meso_div_score'] + 
            weights[2] * row['macro_div_score']
        )
    
    data['weighted_divergence'] = data.apply(apply_regime_weighting, axis=1)
    
    # Apply transaction size momentum adjustment
    data['final_alpha'] = (
        data['weighted_divergence'] * 
        (1 + 0.1 * np.tanh(data['transaction_momentum']))
    )
    
    # Clean and return
    alpha = data['final_alpha'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return alpha
