import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor using multi-timeframe momentum-volume divergence with volatility-regime adjusted weighting
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Calculate returns for volatility calculation
    returns = data['close'].pct_change()
    
    # 1. Multi-Timeframe Momentum Calculations
    # Short-term momentum (5-day ROC)
    mom_5d = (data['close'] / data['close'].shift(5) - 1).fillna(0)
    
    # Medium-term momentum (10-day ROC)
    mom_10d = (data['close'] / data['close'].shift(10) - 1).fillna(0)
    
    # Long-term momentum (20-day ROC)
    mom_20d = (data['close'] / data['close'].shift(20) - 1).fillna(0)
    
    # 2. Volume Confirmation
    # Calculate volume outliers (2x average volume)
    vol_avg_20d = data['volume'].rolling(window=20).mean()
    high_volume_days = (data['volume'] > 2 * vol_avg_20d).astype(int)
    
    # Volume-weighted momentum signals
    vol_weighted_mom_5d = mom_5d * (1 + high_volume_days * 0.5)
    vol_weighted_mom_10d = mom_10d * (1 + high_volume_days * 0.3)
    vol_weighted_mom_20d = mom_20d * (1 + high_volume_days * 0.2)
    
    # 3. Volatility Regime Classification
    # Calculate 20-day rolling volatility
    vol_20d = returns.rolling(window=20).std().fillna(0)
    vol_threshold = vol_20d.quantile(0.7)
    
    # Define volatility regimes
    high_vol_regime = (vol_20d > vol_threshold).astype(int)
    low_vol_regime = (vol_20d <= vol_threshold).astype(int)
    
    # 4. Dynamic Weighting Based on Volatility Regime
    # Base weights
    base_weights = np.array([0.4, 0.35, 0.25])  # [short, medium, long]
    
    # Adjust weights based on volatility regime
    def get_dynamic_weights(regime):
        if regime == 1:  # High volatility
            return np.array([0.2, 0.35, 0.45])  # Reduce short-term, increase long-term
        else:  # Low volatility
            return np.array([0.5, 0.35, 0.15])  # Increase short-term, reduce long-term
    
    # Apply dynamic weighting
    weighted_momentum = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if i < 20:  # Skip initial period for reliable calculations
            weighted_momentum.iloc[i] = 0
            continue
            
        current_weights = get_dynamic_weights(high_vol_regime.iloc[i])
        
        # Combine momentum signals with dynamic weights
        momentum_combined = (
            vol_weighted_mom_5d.iloc[i] * current_weights[0] +
            vol_weighted_mom_10d.iloc[i] * current_weights[1] +
            vol_weighted_mom_20d.iloc[i] * current_weights[2]
        )
        
        weighted_momentum.iloc[i] = momentum_combined
    
    # 5. Signal Validation Framework
    # Price level confirmation (20-day highs/lows)
    high_20d = data['high'].rolling(window=20).max()
    low_20d = data['low'].rolling(window=20).min()
    
    # Breakout vs reversal signals
    near_high = (data['close'] > high_20d * 0.98).astype(int)
    near_low = (data['close'] < low_20d * 1.02).astype(int)
    
    # Volume-price alignment
    price_trend_5d = data['close'].rolling(window=5).apply(
        lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1, raw=False
    ).fillna(0)
    
    volume_trend_5d = data['volume'].rolling(window=5).apply(
        lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1, raw=False
    ).fillna(0)
    
    volume_price_alignment = (price_trend_5d * volume_trend_5d).fillna(0)
    
    # Cross-timeframe consistency
    mom_sign_5d = np.sign(mom_5d)
    mom_sign_10d = np.sign(mom_10d)
    mom_sign_20d = np.sign(mom_20d)
    
    timeframe_agreement = (
        (mom_sign_5d == mom_sign_10d).astype(int) * 0.3 +
        (mom_sign_10d == mom_sign_20d).astype(int) * 0.4 +
        (mom_sign_5d == mom_sign_20d).astype(int) * 0.3
    ).fillna(0)
    
    # 6. Final Factor Construction
    # Apply signal validation adjustments
    validated_factor = weighted_momentum.copy()
    
    # Enhance signals with positive validation
    positive_validation = (
        (near_high & (weighted_momentum > 0)) |
        (near_low & (weighted_momentum < 0)) |
        (volume_price_alignment > 0) |
        (timeframe_agreement > 0.6)
    )
    
    validated_factor[positive_validation] = validated_factor[positive_validation] * 1.2
    
    # Penalize conflicting signals
    negative_validation = (
        (near_high & (weighted_momentum < 0)) |
        (near_low & (weighted_momentum > 0)) |
        (volume_price_alignment < -0.5) |
        (timeframe_agreement < 0.3)
    )
    
    validated_factor[negative_validation] = validated_factor[negative_validation] * 0.7
    
    # In high volatility, require stronger volume confirmation
    high_vol_penalty = (high_vol_regime == 1) & (high_volume_days == 0)
    validated_factor[high_vol_penalty] = validated_factor[high_vol_penalty] * 0.8
    
    return validated_factor.fillna(0)
