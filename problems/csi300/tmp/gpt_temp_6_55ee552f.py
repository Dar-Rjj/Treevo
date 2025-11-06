import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Momentum Elasticity with Fractal Flow Confirmation
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    factor = pd.Series(index=df.index, dtype=float)
    
    # Minimum required data points
    min_periods = 20
    
    for i in range(min_periods, len(df)):
        current_data = df.iloc[:i+1].copy()
        
        # === Fractal Momentum Efficiency Analysis ===
        
        # Multi-Timeframe Momentum Quality
        if i >= 10:
            # Short-term momentum efficiency (3-day)
            close_t = current_data['close'].iloc[i]
            close_t_3 = current_data['close'].iloc[i-3] if i >= 3 else current_data['close'].iloc[0]
            volume_sum_3d = current_data['volume'].iloc[max(0, i-3):i+1].sum()
            momentum_3d = (close_t - close_t_3) / volume_sum_3d if volume_sum_3d > 0 else 0
            
            # Medium-term momentum persistence (10-day)
            close_t_10 = current_data['close'].iloc[i-10] if i >= 10 else current_data['close'].iloc[0]
            volume_sum_10d = current_data['volume'].iloc[max(0, i-10):i+1].sum()
            momentum_10d = (close_t - close_t_10) / volume_sum_10d if volume_sum_10d > 0 else 0
            
            # Momentum decay ratio
            momentum_decay_ratio = momentum_3d / momentum_10d if momentum_10d != 0 else 0
            
            # Momentum quality signals
            momentum_quality = 0
            if momentum_3d > 0 and abs(momentum_decay_ratio) < 2:
                momentum_quality = 1  # Quality persistence
            elif momentum_3d < 0 and volume_sum_3d > volume_sum_10d * 0.8:
                momentum_quality = -1  # Reversal imminent
            elif momentum_3d > 0 and momentum_decay_ratio > 2:
                momentum_quality = -0.5  # Momentum exhaustion
        else:
            momentum_quality = 0
            momentum_3d = 0
            momentum_10d = 0
        
        # === Volatility-Adaptive Price Elasticity ===
        
        # Intraday volatility resilience
        high_t = current_data['high'].iloc[i]
        low_t = current_data['low'].iloc[i]
        close_t = current_data['close'].iloc[i]
        open_t = current_data['open'].iloc[i]
        
        intraday_range = high_t - low_t
        price_change = abs(close_t - open_t)
        
        volatility_resilience = intraday_range / price_change if price_change > 0 else 1
        
        # Price recovery efficiency
        max_drawdown = open_t - low_t if open_t > low_t else 0
        recovery_ratio = (close_t - low_t) / intraday_range if intraday_range > 0 else 0
        
        # Flow-weighted elasticity
        recent_volume = current_data['volume'].iloc[max(0, i-5):i+1].mean()
        avg_volume = current_data['volume'].iloc[max(0, i-20):i+1].mean()
        
        volume_absorption = recent_volume / avg_volume if avg_volume > 0 else 1
        
        # Elasticity signals
        elasticity_signal = 0
        if recovery_ratio > 0.6 and volume_absorption > 1.2:
            elasticity_signal = 1  # Trend resilience
        elif recovery_ratio < 0.4 and volume_absorption < 0.8:
            elasticity_signal = -1  # Fragile structure
        elif recovery_ratio > 0.5 and volume_absorption > 1:
            elasticity_signal = 0.5  # Recovery potential
        
        # === Fractal Flow-Momentum Convergence ===
        
        # Volume-price synchronization
        if i >= 15:
            recent_corr_window = min(10, i)
            price_changes = current_data['close'].iloc[i-recent_corr_window+1:i+1].pct_change().dropna()
            volume_changes = current_data['volume'].iloc[i-recent_corr_window+1:i+1].pct_change().dropna()
            
            if len(price_changes) > 2 and len(volume_changes) > 2:
                volume_price_corr = price_changes.corr(volume_changes)
            else:
                volume_price_corr = 0
        else:
            volume_price_corr = 0
        
        # Volatility regime detection
        recent_volatility = current_data['high'].iloc[max(0, i-5):i+1].std()
        historical_volatility = current_data['high'].iloc[max(0, i-20):i+1].std()
        
        volatility_regime = recent_volatility / historical_volatility if historical_volatility > 0 else 1
        
        # Adaptive convergence signals
        convergence_signal = 0
        
        # High volatility regime
        if volatility_regime > 1.5:
            if elasticity_signal > 0 and momentum_3d > 0:
                convergence_signal = 1
            elif elasticity_signal < 0 or momentum_3d < 0:
                convergence_signal = -1
        
        # Normal volatility regime
        else:
            if momentum_quality > 0 and elasticity_signal > 0 and volume_price_corr > 0.3:
                convergence_signal = 1  # Strong convergence
            elif momentum_quality < 0 and elasticity_signal > 0 and volume_absorption > 1:
                convergence_signal = 0.5  # Quality divergence
            elif momentum_quality < 0 and elasticity_signal < 0 and volume_price_corr < -0.2:
                convergence_signal = -1  # Breakdown pattern
        
        # === Volume Anomaly Validation ===
        
        # Volume persistence and stability
        if i >= 10:
            volume_variance = current_data['volume'].iloc[max(0, i-10):i+1].var()
            volume_mean = current_data['volume'].iloc[max(0, i-10):i+1].mean()
            volume_stability = volume_variance / volume_mean if volume_mean > 0 else 0
            
            # Volume anomaly detection
            current_volume = current_data['volume'].iloc[i]
            volume_anomaly = current_volume / volume_mean if volume_mean > 0 else 1
        else:
            volume_stability = 0
            volume_anomaly = 1
        
        # Final signal confirmation with volume validation
        final_signal = convergence_signal
        
        # Strengthen signal with volume confirmation
        if convergence_signal > 0 and volume_anomaly > 1.2 and volume_price_corr > 0.2:
            final_signal *= 1.2
        elif convergence_signal > 0 and volume_anomaly < 0.8:
            final_signal *= 0.8  # Weaken due to lack of volume confirmation
        
        # Weaken signal with volume contradiction
        if convergence_signal < 0 and volume_anomaly > 1.5 and volume_price_corr < -0.3:
            final_signal *= 1.2  # Strengthen breakdown signal
        elif convergence_signal < 0 and volume_anomaly < 0.7:
            final_signal *= 0.8  # Weaken breakdown signal
        
        # Apply volatility regime weighting
        if volatility_regime > 1.5:
            final_signal *= 0.8  # Reduce confidence in high volatility
        elif volatility_regime < 0.7:
            final_signal *= 1.1  # Increase confidence in low volatility
        
        factor.iloc[i] = final_signal
    
    # Fill initial NaN values with 0
    factor = factor.fillna(0)
    
    return factor
