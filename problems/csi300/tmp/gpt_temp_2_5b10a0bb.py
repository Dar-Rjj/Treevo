import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate novel alpha factor combining volatility-normalized momentum, 
    price-volume efficiency dynamics, and volatility regime adaptation.
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
    # Intraday Price Efficiency
    daily_efficiency = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    efficiency_momentum_5d = daily_efficiency / daily_efficiency.shift(5) - 1
    efficiency_momentum_10d = daily_efficiency / daily_efficiency.shift(10) - 1
    
    # Volume Concentration Metrics
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
            return [0.4, 0.4, 0.15, 0.05]  # High volatility
        elif vol_ratio < 0.67:
            return [0.05, 0.15, 0.4, 0.4]   # Low volatility
        else:
            return [0.2, 0.3, 0.3, 0.2]     # Normal volatility
    
    # Multi-Dimensional Signal Integration
    alpha_scores = pd.Series(index=df.index, dtype=float)
    
    for date in df.index:
        if pd.isna(volatility_ratio.loc[date]) or pd.isna(snr_5d.loc[date]):
            alpha_scores.loc[date] = np.nan
            continue
            
        # Get regime-appropriate weights
        weights = get_regime_weights(volatility_ratio.loc[date])
        
        # Base Momentum Score: Weighted sum of SNR momentum scores
        base_momentum = (weights[0] * momentum_2d.loc[date] + 
                        weights[1] * snr_5d.loc[date] + 
                        weights[2] * snr_10d.loc[date] + 
                        weights[3] * snr_20d.loc[date])
        
        # Cross-Timeframe Momentum Alignment
        momentum_signals = [momentum_2d.loc[date], momentum_5d.loc[date], 
                          momentum_10d.loc[date], momentum_20d.loc[date]]
        directional_consistency = sum(1 for m in momentum_signals if m > 0) / len(momentum_signals)
        
        # Efficiency-Momentum Confirmation
        efficiency_alignment = 1.0
        if (efficiency_momentum_5d.loc[date] > 0 and momentum_5d.loc[date] > 0) or \
           (efficiency_momentum_5d.loc[date] < 0 and momentum_5d.loc[date] < 0):
            efficiency_alignment = 1.2  # Bonus for aligned signals
        elif (efficiency_momentum_5d.loc[date] > 0 and momentum_5d.loc[date] < 0) or \
             (efficiency_momentum_5d.loc[date] < 0 and momentum_5d.loc[date] > 0):
            efficiency_alignment = 0.8  # Penalty for contradictory signals
        
        # Volume Efficiency Confirmation
        volume_confirmation = 1.0
        if not pd.isna(volume_efficiency_momentum.loc[date]):
            if (volume_efficiency_momentum.loc[date] > 0 and momentum_5d.loc[date] > 0) or \
               (volume_efficiency_momentum.loc[date] < 0 and momentum_5d.loc[date] < 0):
                volume_confirmation = 1.1
            elif efficiency_divergence.loc[date] > 0.1:  # Strong divergence
                volume_confirmation = 0.9
        
        # Efficiency Adjustment Factor
        efficiency_stability_factor = 1.0
        if not pd.isna(volume_efficiency_stability.loc[date]) and volume_efficiency_stability.loc[date] > 0:
            efficiency_stability_factor = 1.0 / (1.0 + volume_efficiency_stability.loc[date])
        
        efficiency_adjustment = (efficiency_alignment * volume_confirmation * 
                               efficiency_stability_factor)
        
        # Dynamic scaling based on cross-timeframe agreement
        agreement_multiplier = 0.8 + (directional_consistency * 0.4)
        
        # Final Alpha Factor
        alpha_score = base_momentum * efficiency_adjustment * agreement_multiplier
        
        # Apply regime-specific efficiency emphasis
        if volatility_ratio.loc[date] > 1.5:  # High volatility
            # Focus on recent efficiency changes
            if not pd.isna(efficiency_momentum_5d.loc[date]):
                alpha_score *= (1.0 + 0.3 * np.sign(efficiency_momentum_5d.loc[date]))
        elif volatility_ratio.loc[date] < 0.67:  # Low volatility
            # Emphasize efficiency stability
            if not pd.isna(efficiency_per_vol.loc[date]):
                alpha_score *= (1.0 + 0.2 * np.tanh(efficiency_per_vol.loc[date]))
        
        alpha_scores.loc[date] = alpha_score
    
    return alpha_scores
