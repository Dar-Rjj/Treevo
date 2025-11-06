import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Range-Volume Convergence Factor with Volatility-Adaptive Weighting
    """
    data = df.copy()
    
    # Multi-Period Range Efficiency Calculation
    # Calculate High-Low range
    daily_range = data['high'] - data['low']
    
    # Compute absolute price changes across timeframes
    price_change_5d = (data['close'] - data['close'].shift(5)).abs()
    price_change_20d = (data['close'] - data['close'].shift(20)).abs()
    price_change_60d = (data['close'] - data['close'].shift(60)).abs()
    
    # Calculate Average True Range (ATR) across timeframes
    tr = pd.DataFrame({
        'hl': data['high'] - data['low'],
        'hc': (data['high'] - data['close'].shift(1)).abs(),
        'lc': (data['low'] - data['close'].shift(1)).abs()
    }).max(axis=1)
    
    atr_5d = tr.rolling(window=5).mean()
    atr_20d = tr.rolling(window=20).mean()
    atr_60d = tr.rolling(window=60).mean()
    
    # Compute range efficiency ratios
    efficiency_5d = price_change_5d / atr_5d
    efficiency_20d = price_change_20d / atr_20d
    efficiency_60d = price_change_60d / atr_60d
    
    # Multi-Period Volume Momentum Calculation
    volume_5d_momentum = data['volume'] / data['volume'].shift(5) - 1
    volume_20d_momentum = data['volume'] / data['volume'].shift(20) - 1
    volume_60d_momentum = data['volume'] / data['volume'].shift(60) - 1
    
    # Apply exponential decay weighting for volume momentum
    decay_weights = np.array([0.6, 0.3, 0.1])  # Higher weights to recent periods
    volume_momentum_weighted = (
        volume_5d_momentum * decay_weights[0] + 
        volume_20d_momentum * decay_weights[1] + 
        volume_60d_momentum * decay_weights[2]
    )
    
    # Convergence Pattern Detection
    # Range efficiency convergence
    eff_alignment = (
        np.sign(efficiency_5d - efficiency_5d.shift(1)) + 
        np.sign(efficiency_20d - efficiency_20d.shift(1)) + 
        np.sign(efficiency_60d - efficiency_60d.shift(1))
    ).abs() / 3.0
    
    # Volume momentum convergence
    vol_alignment = (
        np.sign(volume_5d_momentum) + 
        np.sign(volume_20d_momentum) + 
        np.sign(volume_60d_momentum)
    ).abs() / 3.0
    
    # Cross-dimension convergence
    cross_convergence = (
        np.sign(efficiency_5d - efficiency_5d.shift(1)) * np.sign(volume_5d_momentum) + 
        np.sign(efficiency_20d - efficiency_20d.shift(1)) * np.sign(volume_20d_momentum) + 
        np.sign(efficiency_60d - efficiency_60d.shift(1)) * np.sign(volume_60d_momentum)
    ) / 3.0
    
    # Combined convergence score
    convergence_score = (eff_alignment + vol_alignment + (cross_convergence + 1) / 2) / 3
    
    # Volatility Regime Identification
    range_volatility = daily_range.rolling(window=20).std()
    volatility_quantile = range_volatility.rolling(window=60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Volatility regime adjustment factors
    volatility_adjustment = np.where(
        volatility_quantile < 0.3, 1.5,  # Amplify in low volatility
        np.where(volatility_quantile > 0.7, 0.7, 1.0)  # Dampen in high volatility
    )
    
    # Adaptive Convergence Scoring System
    # Volume confirmation weighting
    volume_confirmation = (
        (np.sign(efficiency_5d - efficiency_5d.shift(1)) == np.sign(volume_5d_momentum)).astype(float) * 0.4 +
        (np.sign(efficiency_20d - efficiency_20d.shift(1)) == np.sign(volume_20d_momentum)).astype(float) * 0.35 +
        (np.sign(efficiency_60d - efficiency_60d.shift(1)) == np.sign(volume_60d_momentum)).astype(float) * 0.25
    )
    
    # Multi-Timeframe Factor Construction
    # Combine range efficiency signals
    combined_efficiency = (
        efficiency_5d * 0.5 + 
        efficiency_20d * 0.3 + 
        efficiency_60d * 0.2
    )
    
    # Final factor construction
    alpha_factor = (
        combined_efficiency * 
        convergence_score * 
        (0.6 + 0.4 * volume_confirmation) * 
        volatility_adjustment
    )
    
    return alpha_factor
