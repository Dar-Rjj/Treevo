import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate a novel alpha factor combining volatility-normalized momentum,
    price-volume efficiency dynamics, and regime-adaptive signal integration.
    """
    # Calculate daily returns
    daily_returns = df['close'].pct_change()
    
    # Multi-Timeframe Raw Momentum
    momentum_2d = df['close'] / df['close'].shift(2) - 1
    momentum_5d = df['close'] / df['close'].shift(5) - 1
    momentum_10d = df['close'] / df['close'].shift(10) - 1
    momentum_20d = df['close'] / df['close'].shift(20) - 1
    
    # Realized Volatility by Timeframe
    vol_5d = daily_returns.rolling(window=5).std()
    vol_10d = daily_returns.rolling(window=10).std()
    vol_20d = daily_returns.rolling(window=20).std()
    
    # Signal-to-Noise Ratio Momentum
    snr_5d = momentum_5d / vol_5d.replace(0, np.nan)
    snr_10d = momentum_10d / vol_10d.replace(0, np.nan)
    snr_20d = momentum_20d / vol_20d.replace(0, np.nan)
    
    # Price-Volume Efficiency Dynamics
    daily_efficiency = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    efficiency_momentum_5d = daily_efficiency / daily_efficiency.shift(5) - 1
    efficiency_momentum_10d = daily_efficiency / daily_efficiency.shift(10) - 1
    
    volume_concentration = df['volume'] / (df['high'] - df['low']).replace(0, np.nan)
    amount_concentration = df['amount'] / (df['high'] - df['low']).replace(0, np.nan)
    volume_efficiency_momentum = volume_concentration / volume_concentration.shift(5) - 1
    
    # Efficiency-Volatility Relationship
    efficiency_per_vol = daily_efficiency / vol_5d.replace(0, np.nan)
    volume_efficiency_stability = volume_concentration.rolling(window=5).std()
    efficiency_divergence = efficiency_momentum_5d - volume_efficiency_momentum
    
    # Volatility Regime Classification
    volatility_ratio = vol_5d / vol_20d.replace(0, np.nan)
    
    def get_regime_weights(vol_ratio):
        if vol_ratio > 1.5:
            return [0.4, 0.4, 0.15, 0.05]  # High volatility weights
        elif vol_ratio < 0.67:
            return [0.05, 0.15, 0.4, 0.4]   # Low volatility weights
        else:
            return [0.2, 0.3, 0.3, 0.2]     # Normal volatility weights
    
    # Multi-Dimensional Signal Integration
    alpha_scores = pd.Series(index=df.index, dtype=float)
    
    for date in df.index:
        if pd.isna(volatility_ratio.loc[date]) or pd.isna(snr_5d.loc[date]):
            alpha_scores.loc[date] = np.nan
            continue
            
        # Get regime-appropriate weights
        weights = get_regime_weights(volatility_ratio.loc[date])
        
        # Base Momentum Score
        base_momentum = (weights[0] * momentum_2d.loc[date] + 
                        weights[1] * snr_5d.loc[date] + 
                        weights[2] * snr_10d.loc[date] + 
                        weights[3] * snr_20d.loc[date])
        
        # Directional Consistency Score
        momentum_signals = [momentum_2d.loc[date], momentum_5d.loc[date], 
                          momentum_10d.loc[date], momentum_20d.loc[date]]
        positive_count = sum(1 for sig in momentum_signals if sig > 0)
        directional_score = positive_count / len(momentum_signals)
        
        # Efficiency-Momentum Alignment
        efficiency_alignment = 0
        if not pd.isna(efficiency_momentum_5d.loc[date]):
            if (efficiency_momentum_5d.loc[date] > 0 and momentum_5d.loc[date] > 0) or \
               (efficiency_momentum_5d.loc[date] < 0 and momentum_5d.loc[date] < 0):
                efficiency_alignment = 1
            elif (efficiency_momentum_5d.loc[date] > 0 and momentum_5d.loc[date] < 0) or \
                 (efficiency_momentum_5d.loc[date] < 0 and momentum_5d.loc[date] > 0):
                efficiency_alignment = -1
        
        # Volume Efficiency Confirmation
        volume_confirmation = 0
        if not pd.isna(volume_efficiency_momentum.loc[date]):
            if (volume_efficiency_momentum.loc[date] > 0 and momentum_5d.loc[date] > 0) or \
               (volume_efficiency_momentum.loc[date] < 0 and momentum_5d.loc[date] < 0):
                volume_confirmation = 1
        
        # Efficiency Adjustment Factor
        efficiency_stability_factor = 1.0
        if not pd.isna(volume_efficiency_stability.loc[date]) and volume_efficiency_stability.loc[date] > 0:
            efficiency_stability_factor = 1.0 / (1.0 + volume_efficiency_stability.loc[date])
        
        efficiency_adjustment = (efficiency_alignment + volume_confirmation) * efficiency_stability_factor
        
        # Final Alpha Factor
        cross_timeframe_agreement = directional_score * 2 - 1  # Convert to [-1, 1] range
        final_alpha = base_momentum * (1 + 0.3 * efficiency_adjustment) * (1 + 0.2 * cross_timeframe_agreement)
        
        alpha_scores.loc[date] = final_alpha
    
    return alpha_scores
