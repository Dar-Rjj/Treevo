import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Momentum Acceleration with Regime-Aware Volume-Price Divergence
    """
    # Extract price and volume data
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    amount = df['amount']
    
    # Multi-Timeframe Momentum Calculation
    # Price Momentum Acceleration
    mom_5d = close / close.shift(5) - 1
    mom_10d = close / close.shift(10) - 1
    mom_20d = close / close.shift(20) - 1
    
    # Volume-Price Divergence
    vol_mom_5d = volume / volume.shift(5) - 1
    price_returns_10d = close.pct_change(periods=10)
    volume_changes_10d = volume.pct_change(periods=10)
    
    # Calculate rolling correlation between price returns and volume changes
    price_vol_corr = pd.Series(index=close.index, dtype=float)
    for i in range(10, len(close)):
        if i >= 10:
            price_window = price_returns_10d.iloc[i-9:i+1]
            volume_window = volume_changes_10d.iloc[i-9:i+1]
            valid_mask = (~price_window.isna()) & (~volume_window.isna())
            if valid_mask.sum() >= 5:  # Minimum observations
                price_vol_corr.iloc[i] = np.corrcoef(price_window[valid_mask], 
                                                    volume_window[valid_mask])[0,1]
            else:
                price_vol_corr.iloc[i] = 0
    
    divergence_strength = mom_5d - vol_mom_5d
    
    # Exponential Smoothing Application
    alpha = 0.3
    smooth_mom_5d = mom_5d.ewm(alpha=alpha).mean()
    smooth_mom_10d = mom_10d.ewm(alpha=alpha).mean()
    smooth_mom_20d = mom_20d.ewm(alpha=alpha).mean()
    smooth_divergence = divergence_strength.ewm(alpha=alpha).mean()
    
    # Calculate momentum acceleration (rate of change)
    mom_accel_5d = smooth_mom_5d.diff(3) / 3
    mom_accel_10d = smooth_mom_10d.diff(5) / 5
    mom_accel_20d = smooth_mom_20d.diff(8) / 8
    
    # Second derivative of price momentum
    mom_second_deriv_5d = mom_accel_5d.diff(3) / 3
    
    # Regime Detection Using Amount Data
    amount_ma_20d = amount.rolling(window=20).mean()
    amount_accel = amount / amount.shift(5) - 1
    
    # Amount-based regime classification
    amount_regime = pd.Series(index=amount.index, dtype=str)
    high_participation_threshold = amount_ma_20d.quantile(0.7)
    amount_regime[amount_ma_20d > high_participation_threshold] = 'high'
    amount_regime[amount_ma_20d <= high_participation_threshold] = 'low'
    
    # Volatility Assessment
    daily_range = (high - low) / close
    vol_20d = daily_range.rolling(window=20).mean()
    volatility_regime = pd.Series(index=close.index, dtype=str)
    vol_threshold = vol_20d.quantile(0.7)
    volatility_regime[vol_20d > vol_threshold] = 'volatile'
    volatility_regime[vol_20d <= vol_threshold] = 'stable'
    
    # Cross-Sectional Ranking
    # Calculate cross-sectional ranks for each component
    def cross_sectional_rank(series):
        return series.rank(pct=True)
    
    rank_mom_accel = cross_sectional_rank(mom_accel_5d)
    rank_divergence = cross_sectional_rank(smooth_divergence)
    
    # Volatility normalization
    vol_normalized_mom = mom_accel_5d / (vol_20d + 1e-8)
    vol_normalized_div = smooth_divergence / (vol_20d + 1e-8)
    
    # Adaptive Factor Construction
    # Signal Blending with regime-aware weights
    alpha_signal = pd.Series(index=close.index, dtype=float)
    
    for i in range(20, len(close)):
        if pd.isna(amount_regime.iloc[i]) or pd.isna(volatility_regime.iloc[i]):
            continue
            
        # Base momentum signal
        momentum_signal = 0.4 * rank_mom_accel.iloc[i] + 0.3 * rank_divergence.iloc[i] + 0.3 * vol_normalized_mom.iloc[i]
        
        # Regime-adaptive scaling
        if volatility_regime.iloc[i] == 'volatile':
            # High volatility: emphasize volume confirmation
            vol_weight = 0.6
            mom_weight = 0.4
            if price_vol_corr.iloc[i] > 0.3:  # Strong positive correlation
                regime_multiplier = 1.2
            else:
                regime_multiplier = 0.8
        else:
            # Low volatility: emphasize momentum persistence
            vol_weight = 0.3
            mom_weight = 0.7
            if amount_regime.iloc[i] == 'high':
                regime_multiplier = 1.1
            else:
                regime_multiplier = 0.9
        
        # Multi-timeframe consistency check
        timeframe_consistency = 0
        if (mom_accel_5d.iloc[i] > 0 and mom_accel_10d.iloc[i] > 0 and mom_accel_20d.iloc[i] > 0):
            timeframe_consistency = 0.2
        elif (mom_accel_5d.iloc[i] < 0 and mom_accel_10d.iloc[i] < 0 and mom_accel_20d.iloc[i] < 0):
            timeframe_consistency = -0.2
        
        # Final signal construction
        final_signal = (mom_weight * momentum_signal + 
                       vol_weight * vol_normalized_div.iloc[i] + 
                       timeframe_consistency) * regime_multiplier
        
        # Use amount acceleration for regime transitions
        if abs(amount_accel.iloc[i]) > 0.2:  # Significant regime transition
            final_signal *= (1 + 0.3 * np.sign(amount_accel.iloc[i]))
        
        alpha_signal.iloc[i] = final_signal
    
    # Ensure stationarity by removing extreme values
    alpha_signal = alpha_signal.clip(lower=alpha_signal.quantile(0.01), 
                                   upper=alpha_signal.quantile(0.99))
    
    return alpha_signal
