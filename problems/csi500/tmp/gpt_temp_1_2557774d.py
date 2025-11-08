import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Volume-Weighted Reversal factor
    """
    # Extract price and volume data
    close = df['close']
    volume = df['volume']
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate returns for volatility and correlation calculations
    returns = close.pct_change()
    volume_changes = volume.pct_change()
    
    for i in range(20, len(df)):
        current_date = df.index[i]
        
        # Volatility-Normalized Reversal Component
        # Short-Term Reversal (3-day)
        if i >= 6:
            short_raw_reversal = (close.iloc[i-1] - close.iloc[i-3]) / close.iloc[i-3]
            # 5-day rolling volatility (using returns from t-6 to t-1)
            short_vol = returns.iloc[i-5:i].std()
            short_norm_reversal = short_raw_reversal / short_vol if short_vol > 0 else 0
        else:
            short_norm_reversal = 0
        
        # Medium-Term Reversal (8-day)
        if i >= 11:
            medium_raw_reversal = (close.iloc[i-1] - close.iloc[i-8]) / close.iloc[i-8]
            # 10-day rolling volatility (using returns from t-11 to t-1)
            medium_vol = returns.iloc[i-10:i].std()
            medium_norm_reversal = medium_raw_reversal / medium_vol if medium_vol > 0 else 0
        else:
            medium_norm_reversal = 0
        
        # Multi-Timeframe Signal Blending
        blended_reversal = 0.6 * short_norm_reversal + 0.4 * medium_norm_reversal
        
        # Adaptive Threshold Application
        if i >= 21:
            threshold = np.percentile([blended_reversal] + [0] * 19, 20)  # Simplified percentile
            reversal_signal = 1 if blended_reversal > threshold else -1
        else:
            reversal_signal = 1
        
        # Volume-Price Divergence Analysis
        # Nonlinear Volume Weighting
        if i >= 8:
            vol_3d_change = volume.iloc[i] / volume.iloc[i-3] - 1 if volume.iloc[i-3] > 0 else 0
            vol_8d_change = volume.iloc[i] / volume.iloc[i-8] - 1 if volume.iloc[i-8] > 0 else 0
            raw_divergence = vol_3d_change - vol_8d_change
            nonlinear_vol_weight = np.sign(raw_divergence) * np.sqrt(np.abs(raw_divergence)) if np.abs(raw_divergence) > 0 else 0
        else:
            nonlinear_vol_weight = 0
        
        # Price-Volume Relationship Strength
        if i >= 6:
            # 5-day rolling correlation
            price_changes = returns.iloc[i-5:i]
            volume_changes_5d = volume_changes.iloc[i-5:i]
            if len(price_changes) >= 3 and len(volume_changes_5d) >= 3:
                correlation = price_changes.corr(volume_changes_5d)
                correlation_magnitude = abs(correlation) if not np.isnan(correlation) else 0
            else:
                correlation_magnitude = 0
        else:
            correlation_magnitude = 0
        
        # Volume Regime Detection
        if i >= 20:
            # Volume Spike Detection
            vol_sma_10 = volume.iloc[i-10:i].mean()
            vol_std_10 = volume.iloc[i-10:i].std()
            vol_zscore = (volume.iloc[i] - vol_sma_10) / vol_std_10 if vol_std_10 > 0 else 0
            spike_regime = vol_zscore > 2
            
            # Volume Trend Regime
            vol_sma_5 = volume.iloc[i-4:i+1].mean() if i >= 4 else volume.iloc[:i+1].mean()
            vol_sma_20 = volume.iloc[i-19:i+1].mean() if i >= 19 else volume.iloc[:i+1].mean()
            trend_signal = 1 if vol_sma_5 > vol_sma_20 else -1
            
            # Combined Volume Signal
            if spike_regime:
                volume_weight = nonlinear_vol_weight * 2
            else:
                volume_weight = nonlinear_vol_weight
        else:
            volume_weight = nonlinear_vol_weight
            trend_signal = 1
        
        # Regime-Based Signal Selection
        # Volatility Regime Detection
        if i >= 20:
            vol_5d = returns.iloc[i-4:i+1].std() if i >= 4 else returns.iloc[:i+1].std()
            vol_20d = returns.iloc[i-19:i+1].std() if i >= 19 else returns.iloc[:i+1].std()
            high_vol_regime = vol_5d > vol_20d
            
            if high_vol_regime:
                regime_selected_reversal = short_norm_reversal
            else:
                regime_selected_reversal = medium_norm_reversal
        else:
            regime_selected_reversal = blended_reversal
        
        # Factor Combination
        base_factor = regime_selected_reversal * volume_weight
        
        # Correlation Adjustment
        if correlation_magnitude > 0.3:  # Strong correlation
            if returns.iloc[i] > 0 and volume.iloc[i] > volume.iloc[i-1]:  # Positive relationship
                final_factor = base_factor
            else:  # Negative relationship
                final_factor = -base_factor
        elif correlation_magnitude < 0.1:  # Weak correlation
            final_factor = base_factor * correlation_magnitude
        else:  # Moderate correlation
            final_factor = base_factor
        
        result.iloc[i] = final_factor * reversal_signal * trend_signal
    
    # Fill NaN values with 0
    result = result.fillna(0)
    
    return result
