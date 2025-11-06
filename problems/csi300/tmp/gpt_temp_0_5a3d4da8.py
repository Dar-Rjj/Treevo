import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum-Reversal Divergence Calculation
    # Short-Term Momentum and Reversal
    short_momentum = data['close'] / data['close'].shift(5) - 1
    short_reversal = data['close'] / data['close'].shift(1) - 1
    
    # Medium-Term Momentum and Reversal
    medium_momentum = data['close'] / data['close'].shift(20) - 1
    medium_reversal = data['close'] / data['close'].shift(10) - 1
    
    # Calculate divergences
    short_divergence = short_momentum - short_reversal
    medium_divergence = medium_momentum - medium_reversal
    
    # Combined divergence signal
    divergence_signal = 0.6 * short_divergence + 0.4 * medium_divergence
    
    # Volatility-Weighted Adjustment
    # Calculate daily range and 10-day average range
    daily_range = data['high'] - data['low']
    avg_range_10d = daily_range.rolling(window=10, min_periods=10).mean()
    
    # Apply volatility scaling with dampening
    volatility_adjusted = divergence_signal / (avg_range_10d + 1e-8)
    volatility_dampened = volatility_adjusted / (1 + np.abs(volatility_adjusted))
    
    # Liquidity Confirmation with Volume Momentum
    # Amount-based liquidity assessment
    median_amount_15d = data['amount'].rolling(window=15, min_periods=15).median()
    amount_ratio = data['amount'] / (median_amount_15d + 1e-8)
    log_amount_ratio = np.log1p(np.abs(amount_ratio - 1)) * np.sign(amount_ratio - 1)
    
    # Volume momentum analysis
    volume_5d_avg = data['volume'].rolling(window=5, min_periods=5).mean()
    volume_20d_avg = data['volume'].rolling(window=20, min_periods=20).mean()
    volume_momentum_ratio = volume_5d_avg / (volume_20d_avg + 1e-8)
    
    # Apply consistency filter to volume momentum
    volume_consistency = np.where(
        (volume_momentum_ratio > 0.8) & (volume_momentum_ratio < 1.2),
        volume_momentum_ratio,
        1.0
    )
    
    # Integrated Factor Generation
    # Combine divergence with liquidity signals
    combined_factor = (
        volatility_dampened * 
        (1 + 0.1 * log_amount_ratio) * 
        volume_consistency
    )
    
    # Price Range Stability Adjustment
    avg_range_20d = daily_range.rolling(window=20, min_periods=20).mean()
    range_stability = daily_range / (avg_range_20d + 1e-8)
    range_stability_score = 1 / (1 + np.abs(range_stability - 1))
    
    # Apply range stability weight
    stability_adjusted = combined_factor * range_stability_score
    
    # Time Persistence Enhancement
    # Calculate persistence score over 5 days
    factor_direction = np.sign(stability_adjusted)
    persistence_score = factor_direction.rolling(window=5, min_periods=5).apply(
        lambda x: np.sum(x == x.iloc[-1]) / len(x) if len(x) > 0 else 1.0
    )
    
    # Apply persistence multiplier
    final_factor = stability_adjusted * persistence_score
    
    return final_factor
