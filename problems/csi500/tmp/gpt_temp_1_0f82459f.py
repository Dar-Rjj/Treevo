import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Regime-Aware Momentum Acceleration factor
    Combines momentum acceleration with amount-based regime detection and volatility scaling
    """
    df = data.copy()
    
    # Momentum Acceleration Calculation
    # Exponential smoothing of price returns
    returns = df['close'].pct_change()
    momentum = returns.ewm(alpha=0.3, adjust=False).mean()
    momentum_acceleration = momentum.diff().ewm(alpha=0.3, adjust=False).mean()
    
    # Regime Detection Using Amount Data
    amount_20d_avg = df['amount'].rolling(window=20).mean()
    amount_acceleration = amount_20d_avg.pct_change(periods=5)
    
    # Regime classification
    high_participation_regime = amount_acceleration > 0.1
    low_participation_regime = amount_acceleration <= 0.1
    
    # Volatility Scaling
    daily_range = (df['high'] - df['low']) / df['close']
    volatility_20d = daily_range.rolling(window=20).mean()
    
    # Volatility-adjusted momentum acceleration
    volatility_adjusted_acceleration = momentum_acceleration / (volatility_20d + 1e-8)
    
    # Cross-sectional ranking (within each date)
    def cross_sectional_rank(series):
        return series.rank(pct=True)
    
    ranked_acceleration = volatility_adjusted_acceleration.groupby(level='date').transform(cross_sectional_rank)
    
    # Regime-adaptive weighting
    base_signal = ranked_acceleration
    
    # High participation: emphasize recent acceleration (higher weight)
    high_regime_weight = 1.2
    # Low participation: emphasize momentum persistence (lower weight)
    low_regime_weight = 0.8
    
    # Apply regime-specific weights
    regime_weighted_signal = base_signal.copy()
    regime_weighted_signal[high_participation_regime] = base_signal[high_participation_regime] * high_regime_weight
    regime_weighted_signal[low_participation_regime] = base_signal[low_participation_regime] * low_regime_weight
    
    # Final factor construction
    final_factor = regime_weighted_signal
    
    return final_factor
