import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Multi-Timeframe Momentum Acceleration with Regime-Aware Volume-Price Divergence
    
    This alpha factor combines:
    - Multi-timeframe momentum acceleration (5, 10, 20 days)
    - Volume-price divergence analysis
    - Amount-based regime detection
    - Exponential smoothing for signal stability
    - Cross-sectional ranking and volatility adjustment
    """
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    # Required minimum data points
    min_periods = 20
    
    for i in range(min_periods, len(data)):
        current_data = data.iloc[:i+1]
        
        # Multi-Timeframe Momentum Acceleration
        # Price Momentum Calculation
        if i >= 5:
            momentum_5d = (current_data['close'].iloc[i] / current_data['close'].iloc[i-5]) - 1
        else:
            momentum_5d = 0
            
        if i >= 10:
            momentum_10d = (current_data['close'].iloc[i] / current_data['close'].iloc[i-10]) - 1
        else:
            momentum_10d = 0
            
        if i >= 20:
            momentum_20d = (current_data['close'].iloc[i] / current_data['close'].iloc[i-20]) - 1
        else:
            momentum_20d = 0
        
        # Momentum Acceleration Derivation
        if i >= 10:
            accel_5_10 = momentum_10d - momentum_5d
        else:
            accel_5_10 = 0
            
        if i >= 20:
            accel_10_20 = momentum_20d - momentum_10d
        else:
            accel_10_20 = 0
        
        # Composite acceleration score (weighted average)
        composite_accel = 0.4 * accel_5_10 + 0.6 * accel_10_20
        
        # Volume-Price Divergence Analysis
        # Volume-Adjusted Price Movement
        if i >= 1:
            volume_price_change = (current_data['close'].iloc[i] - current_data['close'].iloc[i-1]) * current_data['volume'].iloc[i]
        else:
            volume_price_change = 0
        
        # 5-day cumulative volume-price
        if i >= 5:
            cum_volume_price = 0
            for j in range(max(1, i-4), i+1):
                price_change = current_data['close'].iloc[j] - current_data['close'].iloc[j-1]
                cum_volume_price += price_change * current_data['volume'].iloc[j]
        else:
            cum_volume_price = 0
        
        # Normalized volume-price relationship
        if i >= 5 and current_data['volume'].iloc[i-4:i+1].sum() > 0:
            volume_price_ratio = cum_volume_price / current_data['volume'].iloc[i-4:i+1].sum()
        else:
            volume_price_ratio = 0
        
        # Divergence Detection
        volume_trend = volume_price_ratio
        price_trend = composite_accel
        
        # Divergence score: positive when volume confirms price, negative when diverges
        if abs(price_trend) > 0.001:
            divergence_score = volume_trend / abs(price_trend) if abs(price_trend) > 0 else 0
        else:
            divergence_score = volume_trend * 1000  # Scale for small price movements
        
        # Regime Detection Using Amount Data
        # Amount-Based Market Participation
        if i >= 20:
            amount_ma_20d = current_data['amount'].iloc[i-19:i+1].mean()
        else:
            amount_ma_20d = current_data['amount'].iloc[:i+1].mean()
        
        if i >= 5:
            amount_accel = (current_data['amount'].iloc[i] / current_data['amount'].iloc[i-5]) - 1
        else:
            amount_accel = 0
        
        # Regime classification
        amount_threshold = amount_ma_20d * 1.1  # 10% above 20-day MA
        high_participation = current_data['amount'].iloc[i] > amount_threshold
        
        # Exponential Smoothing Application (α=0.3)
        alpha = 0.3
        
        # Initialize smoothed values if first calculation
        if i == min_periods:
            smoothed_accel = composite_accel
            smoothed_divergence = divergence_score
            smoothed_regime = 1.0 if high_participation else 0.0
        else:
            # Adaptive smoothing based on regime volatility
            if high_participation:
                # More responsive in high participation regimes
                adaptive_alpha = min(alpha * 1.5, 0.8)
            else:
                # More smoothing in low participation
                adaptive_alpha = max(alpha * 0.7, 0.1)
            
            smoothed_accel = adaptive_alpha * composite_accel + (1 - adaptive_alpha) * result.iloc[i-1]
            smoothed_divergence = adaptive_alpha * divergence_score + (1 - adaptive_alpha) * result.iloc[i-1]
            smoothed_regime = adaptive_alpha * (1.0 if high_participation else 0.0) + (1 - adaptive_alpha) * result.iloc[i-1]
        
        # Cross-Sectional Ranking & Volatility Adjustment
        # 20-day price range volatility
        if i >= 20:
            high_20d = current_data['high'].iloc[i-19:i+1].max()
            low_20d = current_data['low'].iloc[i-19:i+1].min()
            close_20d = current_data['close'].iloc[i-19:i+1].mean()
            if close_20d > 0:
                volatility_20d = (high_20d - low_20d) / close_20d
            else:
                volatility_20d = 0
        else:
            volatility_20d = 0.1  # Default low volatility
        
        # Regime-Adaptive Factor Construction
        # Signal Combination Logic
        if smoothed_accel > 0 and smoothed_divergence > 0:
            # Bullish: positive acceleration + volume confirmation
            base_signal = smoothed_accel * (1 + smoothed_divergence)
        elif smoothed_accel < 0 and smoothed_divergence < 0:
            # Bearish: negative acceleration + volume divergence
            base_signal = smoothed_accel * (1 - smoothed_divergence)
        else:
            # Neutral: conflicting signals
            base_signal = smoothed_accel * 0.5
        
        # Regime-Weighted Integration
        if high_participation:
            # High participation: emphasize volume confirmation
            regime_weighted_signal = base_signal * (1 + 0.3 * smoothed_divergence)
        else:
            # Low participation: emphasize price momentum
            regime_weighted_signal = base_signal * (1 + 0.3 * smoothed_accel)
        
        # Volatility normalization
        if volatility_20d > 0:
            volatility_adjusted_signal = regime_weighted_signal / volatility_20d
        else:
            volatility_adjusted_signal = regime_weighted_signal
        
        # Final Alpha Output
        result.iloc[i] = volatility_adjusted_signal
    
    # Fill initial values with 0
    result.iloc[:min_periods] = 0
    
    return result
