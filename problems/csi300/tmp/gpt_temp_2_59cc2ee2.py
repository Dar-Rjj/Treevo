import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Momentum Acceleration with Volume-Volatility Confirmation factor
    """
    data = df.copy()
    
    # Multi-Timeframe Momentum Analysis
    data['ret_3d'] = data['close'] / data['close'].shift(3) - 1
    data['ret_8d'] = data['close'] / data['close'].shift(8) - 1
    data['ret_21d'] = data['close'] / data['close'].shift(21) - 1
    
    # Momentum acceleration differences
    data['mom_accel_3_8'] = data['ret_3d'] - data['ret_8d']
    data['mom_accel_8_21'] = data['ret_8d'] - data['ret_21d']
    
    # Momentum curvature (second-order differences)
    data['mom_curvature'] = data['mom_accel_3_8'] - data['mom_accel_8_21']
    
    # Volume and Amount Confirmation
    data['vol_mom_5d'] = data['volume'] / data['volume'].shift(5) - 1
    data['vol_mom_10d'] = data['volume'] / data['volume'].shift(10) - 1
    data['vol_accel'] = data['vol_mom_5d'] - data['vol_mom_10d']
    
    # Price-volume alignment
    data['price_vol_alignment'] = np.sign(data['ret_8d']) * np.sign(data['vol_mom_5d'])
    
    # Amount-based metrics
    data['amount_mom_5d'] = data['amount'] / data['amount'].shift(5) - 1
    data['vol_to_amount_ratio'] = data['volume'] / data['amount']
    data['vol_to_amount_mom'] = data['vol_to_amount_ratio'] / data['vol_to_amount_ratio'].shift(5) - 1
    
    # Volatility Context Assessment
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['volatility_5d'] = data['daily_range'].rolling(window=5).mean()
    data['volatility_10d'] = data['daily_range'].rolling(window=10).mean()
    
    # Volatility momentum and acceleration
    data['vol_mom'] = data['volatility_5d'] / data['volatility_10d'] - 1
    data['vol_accel'] = data['volatility_5d'].pct_change(3)
    
    # Volatility regime classification
    data['vol_regime'] = pd.cut(data['volatility_10d'].rolling(window=21).rank(pct=True), 
                               bins=[0, 0.3, 0.7, 1.0], labels=[0, 1, 2])
    
    # Volatility-adjusted momentum
    data['vol_adj_momentum'] = data['ret_8d'] / (data['volatility_5d'] + 1e-8)
    
    # Pattern Recognition
    # Momentum-volume divergence
    data['mom_vol_divergence'] = np.where(
        np.sign(data['ret_8d']) != np.sign(data['vol_mom_5d']),
        np.abs(data['ret_8d']) * np.abs(data['vol_mom_5d']), 0
    )
    
    # Multi-timeframe momentum consistency
    data['momentum_consistency'] = (
        np.sign(data['ret_3d']) * np.sign(data['ret_8d']) * np.sign(data['ret_21d'])
    )
    
    # Momentum regime transitions
    data['momentum_regime_change'] = (
        (data['ret_8d'] > data['ret_8d'].shift(1)) & 
        (data['ret_8d'].shift(1) < data['ret_8d'].shift(2))
    ).astype(int) - (
        (data['ret_8d'] < data['ret_8d'].shift(1)) & 
        (data['ret_8d'].shift(1) > data['ret_8d'].shift(2))
    ).astype(int)
    
    # Composite Factor Generation
    # Base momentum acceleration factor
    base_factor = (
        data['mom_curvature'] * 0.4 +
        data['vol_adj_momentum'] * 0.3 +
        data['price_vol_alignment'] * 0.3
    )
    
    # Volume confirmation multiplier
    volume_confirmation = (
        np.tanh(data['vol_accel'] * 2) * 0.6 +
        np.tanh(data['vol_to_amount_mom'] * 3) * 0.4
    )
    
    # Pattern recognition signals
    pattern_signals = (
        (1 + data['momentum_consistency'] * 0.2) *
        (1 - np.tanh(np.abs(data['mom_vol_divergence']) * 5) * 0.15) *
        (1 + data['momentum_regime_change'] * 0.1)
    )
    
    # Volatility regime weighting
    regime_weights = {
        0: 1.2,  # Low volatility - higher confidence
        1: 1.0,  # Normal volatility - standard weight
        2: 0.7   # High volatility - reduced confidence
    }
    regime_multiplier = data['vol_regime'].map(regime_weights)
    
    # Final composite factor
    final_factor = (
        base_factor * 
        (1 + volume_confirmation) * 
        pattern_signals * 
        regime_multiplier
    )
    
    # Clean and return
    final_factor = final_factor.replace([np.inf, -np.inf], np.nan)
    final_factor = final_factor.fillna(0)
    
    return final_factor
