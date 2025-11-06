import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Intraday Momentum Reversal Divergence
    df = df.copy()
    
    # Calculate intraday momentum
    intraday_momentum = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    momentum_sign = np.sign(intraday_momentum)
    
    # Detect reversal patterns
    prev_momentum_sign = momentum_sign.shift(1)
    momentum_divergence = momentum_sign != prev_momentum_sign
    
    # Volume confirmation
    volume_ratio = df['volume'] / (df['volume'].shift(1) + 1e-8)
    reversal_factor = momentum_divergence.astype(float) * intraday_momentum * volume_ratio
    
    # Volatility-Clustered Price Impact
    # Identify volatility clusters
    daily_range = df['high'] - df['low']
    vol_quantile = daily_range.rolling(window=20, min_periods=10).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # Price impact efficiency
    price_impact = abs(df['close'] - df['open']) / (df['amount'] + 1e-8)
    vol_cluster_factor = price_impact * vol_quantile
    
    # Amount-Volume Divergence Oscillator
    # Normalized amount flow
    amount_flow = df['amount'] / (daily_range + 1e-8)
    
    # Volume-amount divergence
    volume_change = df['volume'] / (df['volume'].shift(1) + 1e-8)
    amount_flow_ratio = amount_flow / (amount_flow.shift(1) + 1e-8)
    divergence_oscillator = volume_change / (amount_flow_ratio + 1e-8)
    
    # Open-Close Relative Strength Persistence
    # Daily strength score
    strength_score = (df['close'] - df['open']) / (abs(daily_range) + 1e-8)
    strength_sign = np.sign(strength_score)
    
    # Persistence pattern
    sign_changes = strength_sign != strength_sign.shift(1)
    persistence_count = sign_changes.rolling(window=5).apply(lambda x: 5 - sum(x), raw=True)
    avg_strength = abs(strength_score).rolling(window=5).mean()
    persistence_factor = persistence_count * avg_strength
    
    # High-Frequency Rejection Signal
    # Detect price rejections
    upper_rejection = (df['high'] - np.maximum(df['open'], df['close'])) / (daily_range + 1e-8)
    lower_rejection = (np.minimum(df['open'], df['close']) - df['low']) / (daily_range + 1e-8)
    
    # Combine rejection signals
    rejection_product = upper_rejection * lower_rejection
    volume_weight = df['volume'] / df['volume'].rolling(window=20).mean()
    rejection_signal = rejection_product * volume_weight
    
    # Volatility-Adjusted Amount Efficiency
    # Volatility-normalized amount
    vol_normalized_amount = df['amount'] / (daily_range + 1e-8)
    
    # Efficiency ratio
    price_change = abs(df['close'] - df['close'].shift(1))
    efficiency_ratio = price_change / (vol_normalized_amount + 1e-8)
    
    # Combine all factors with weights
    factor = (
        0.25 * reversal_factor.fillna(0) +
        0.20 * vol_cluster_factor.fillna(0) +
        0.15 * divergence_oscillator.fillna(0) +
        0.15 * persistence_factor.fillna(0) +
        0.15 * rejection_signal.fillna(0) +
        0.10 * efficiency_ratio.fillna(0)
    )
    
    return factor
