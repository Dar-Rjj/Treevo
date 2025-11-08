import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Momentum-Volume Convergence Factor
    Combines price momentum, volume momentum, and market regime detection
    to generate adaptive trading signals
    """
    data = df.copy()
    
    # Price Momentum Calculation
    data['mom_5'] = data['close'] / data['close'].shift(5) - 1
    data['mom_10'] = data['close'] / data['close'].shift(10) - 1
    data['mom_20'] = data['close'] / data['close'].shift(20) - 1
    
    # Volume Momentum Calculation
    data['vol_mom_5'] = data['volume'] / data['volume'].shift(5) - 1
    data['vol_mom_10'] = data['volume'] / data['volume'].shift(10) - 1
    data['vol_mom_20'] = data['volume'] / data['volume'].shift(20) - 1
    
    # Regime Detection System
    data['amount_ma_20'] = data['amount'].rolling(window=20).mean()
    data['amount_accel'] = (data['amount'] / data['amount'].shift(5)) - (data['amount'].shift(5) / data['amount'].shift(10))
    
    # Volatility Environment Assessment
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['volatility_20'] = data['true_range'].rolling(window=20).mean()
    
    # Regime Classification
    amount_threshold = data['amount_ma_20'].quantile(0.7)
    vol_threshold = data['volatility_20'].quantile(0.7)
    
    data['high_participation'] = (data['amount'] > amount_threshold).astype(int)
    data['high_volatility'] = (data['volatility_20'] > vol_threshold).astype(int)
    
    # Adaptive Smoothing Framework
    alpha = 0.3
    data['ema_mom_5'] = data['mom_5'].ewm(alpha=alpha).mean()
    data['ema_vol_mom_5'] = data['vol_mom_5'].ewm(alpha=alpha).mean()
    
    # Momentum Acceleration Calculation
    data['mom_accel'] = data['ema_mom_5'] - data['ema_mom_5'].shift(1)
    data['vol_mom_accel'] = data['ema_vol_mom_5'] - data['ema_vol_mom_5'].shift(1)
    
    # Cross-Sectional Processing (within-day ranking)
    def cross_sectional_processing(group):
        group['mom_divergence'] = group['ema_mom_5'] - group['ema_vol_mom_5']
        group['mom_divergence_rank'] = group['mom_divergence'].rank(pct=True)
        group['vol_mom_rank'] = group['ema_vol_mom_5'].rank(pct=True)
        group['price_mom_rank'] = group['ema_mom_5'].rank(pct=True)
        return group
    
    # Apply cross-sectional processing
    data = data.groupby(data.index).apply(cross_sectional_processing)
    
    # Dynamic Weighting Scheme
    data['volume_weight'] = np.where(data['high_volatility'] == 1, 0.7, 0.3)
    data['price_weight'] = 1 - data['volume_weight']
    
    # Signal Combination Logic
    bullish_condition = (data['ema_mom_5'] > data['ema_vol_mom_5']) & (data['mom_accel'] > 0)
    bearish_condition = (data['ema_mom_5'] < data['ema_vol_mom_5']) & (data['mom_accel'] < 0)
    
    # Base divergence signal
    data['base_signal'] = data['ema_mom_5'] - data['ema_vol_mom_5']
    
    # Regime-Adaptive Scaling
    high_part_multiplier = np.where(data['high_participation'] == 1, 1.5, 0.8)
    regime_adjustment = data['amount_accel'] * 0.1  # Small adjustment for regime transitions
    
    # Final Factor Construction
    data['factor'] = (
        (data['base_signal'] * data['price_weight'] * high_part_multiplier) +
        (data['base_signal'] * data['volume_weight'] * (2 - high_part_multiplier)) +
        regime_adjustment
    )
    
    # Enhance signals with acceleration
    data['factor'] = np.where(bullish_condition, data['factor'] * 1.2, data['factor'])
    data['factor'] = np.where(bearish_condition, data['factor'] * 0.8, data['factor'])
    
    # Stationary measure - normalize by recent volatility
    factor_vol = data['factor'].rolling(window=20).std()
    data['final_factor'] = data['factor'] / (factor_vol + 1e-8)
    
    return data['final_factor']
