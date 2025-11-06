import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum Decay Adjusted by Volume Skewness
    # Calculate Price Momentum with exponential decay
    momentum_window = 20
    decay_rate = 0.9
    
    # Exponential weights for momentum calculation
    weights = np.array([decay_rate ** i for i in range(momentum_window)])[::-1]
    weights = weights / weights.sum()
    
    # Calculate momentum with exponential decay
    momentum = data['close'].rolling(window=momentum_window).apply(
        lambda x: np.sum(weights * (x - x.iloc[0]) / x.iloc[0]), raw=False
    )
    
    # Calculate Volume Skewness with recent weighting
    volume_skew_window = 15
    volume_weights = np.array([1.2 ** i for i in range(volume_skew_window)])[::-1]
    volume_weights = volume_weights / volume_weights.sum()
    
    def weighted_skew(x):
        if len(x) < 3:
            return 0
        weighted_mean = np.sum(volume_weights * x) / volume_weights.sum()
        weighted_std = np.sqrt(np.sum(volume_weights * (x - weighted_mean) ** 2) / volume_weights.sum())
        if weighted_std == 0:
            return 0
        weighted_skewness = np.sum(volume_weights * ((x - weighted_mean) / weighted_std) ** 3) / volume_weights.sum()
        return weighted_skewness
    
    volume_skewness = data['volume'].rolling(window=volume_skew_window).apply(weighted_skew, raw=False)
    
    # Volatility adjustment
    volatility = data['close'].pct_change().rolling(window=10).std()
    
    # Combine signals
    momentum_volume_factor = momentum * volume_skewness
    momentum_volume_factor = momentum_volume_factor / (volatility + 1e-8)
    
    # High-Low Range Breakout Efficiency
    # Calculate True Range
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - data['close'].shift(1))
    tr3 = abs(data['low'] - data['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate Price Movement Efficiency
    price_change = abs(data['close'] - data['close'].shift(5))
    efficiency_ratio = price_change / (true_range.rolling(window=5).sum() + 1e-8)
    
    # Volume Confirmation
    volume_ma = data['volume'].rolling(window=10).mean()
    volume_signal = data['volume'] / (volume_ma + 1e-8)
    
    breakout_efficiency_factor = efficiency_ratio * volume_signal
    
    # Volatility Regime Adjusted Price Reversal
    # Identify Volatility Regime
    volatility_regime = data['close'].pct_change().rolling(window=20).std()
    high_vol_threshold = volatility_regime.quantile(0.7)
    low_vol_threshold = volatility_regime.quantile(0.3)
    
    # Calculate Short-Term Reversal
    short_term_return = data['close'].pct_change(periods=3)
    
    # Adjust for volatility regime
    volatility_adjustment = np.where(
        volatility_regime > high_vol_threshold, 0.7,
        np.where(volatility_regime < low_vol_threshold, 1.3, 1.0)
    )
    
    reversal_signal = -short_term_return * volatility_adjustment
    
    # Volume Acceleration Filter
    volume_acceleration = data['volume'].pct_change(periods=5)
    volume_filter = np.where(volume_acceleration > 0, 1.2, 0.8)
    
    reversal_factor = reversal_signal * volume_filter
    
    # Liquidity-Adjusted Momentum Persistence
    # Compute Momentum Strength
    momentum_5 = data['close'].pct_change(periods=5)
    momentum_10 = data['close'].pct_change(periods=10)
    momentum_consistency = np.sign(momentum_5) * np.sign(momentum_10) * (abs(momentum_5) + abs(momentum_10)) / 2
    
    # Assess Liquidity Conditions
    volume_ma_20 = data['volume'].rolling(window=20).mean()
    amount_ma_20 = data['amount'].rolling(window=20).mean()
    
    volume_liquidity = data['volume'] / (volume_ma_20 + 1e-8)
    amount_liquidity = data['amount'] / (amount_ma_20 + 1e-8)
    
    liquidity_score = (volume_liquidity + amount_liquidity) / 2
    
    # Apply persistence weighting
    momentum_persistence = momentum_consistency.rolling(window=10).apply(
        lambda x: len([i for i in range(1, len(x)) if np.sign(x.iloc[i]) == np.sign(x.iloc[i-1])]) / (len(x)-1) 
        if len(x) > 1 else 0.5, raw=False
    )
    
    liquidity_momentum_factor = momentum_consistency * liquidity_score * momentum_persistence
    
    # Opening Gap Convergence Divergence
    # Calculate Opening Gaps
    opening_gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Analyze Intraday Recovery
    gap_filling = np.where(
        opening_gap > 0,  # Positive gap
        (data['low'] - data['open']) / (data['close'].shift(1) - data['open'] + 1e-8),
        np.where(
            opening_gap < 0,  # Negative gap
            (data['high'] - data['open']) / (data['close'].shift(1) - data['open'] + 1e-8),
            0
        )
    )
    
    convergence_strength = 1 - abs(gap_filling)
    
    # Volume Profile Analysis
    opening_volume_ratio = data['volume'] / data['volume'].rolling(window=10).mean()
    volume_weight = np.where(opening_volume_ratio > 1.2, 1.5, 
                           np.where(opening_volume_ratio < 0.8, 0.7, 1.0))
    
    gap_convergence_factor = opening_gap * convergence_strength * volume_weight
    
    # Combine all factors with equal weighting
    final_factor = (
        momentum_volume_factor.fillna(0) * 0.2 +
        breakout_efficiency_factor.fillna(0) * 0.2 +
        reversal_factor.fillna(0) * 0.2 +
        liquidity_momentum_factor.fillna(0) * 0.2 +
        gap_convergence_factor.fillna(0) * 0.2
    )
    
    return final_factor
