import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor using Price-Volume Divergence Framework with Regime-Adaptive Weighting and Volume Outlier Enhancement
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    factor_values = pd.Series(index=data.index, dtype=float)
    
    # Calculate all required rolling windows
    for i in range(20, len(data)):
        current_data = data.iloc[:i+1]
        
        # Multi-Timeframe Price Momentum
        # Short-Term (3-day)
        if i >= 3:
            close_t = current_data['close'].iloc[i]
            close_t3 = current_data['close'].iloc[i-3]
            high_t2_t = current_data['high'].iloc[i-2:i+1].max()
            low_t2_t = current_data['low'].iloc[i-2:i+1].min()
            
            short_raw_return = close_t / close_t3 - 1
            short_range_adj_return = (close_t - close_t3) / (high_t2_t - low_t2_t) if (high_t2_t - low_t2_t) > 0 else 0
        else:
            short_raw_return = 0
            short_range_adj_return = 0
        
        # Medium-Term (10-day)
        if i >= 10:
            close_t10 = current_data['close'].iloc[i-10]
            high_t9_t = current_data['high'].iloc[i-9:i+1].max()
            low_t9_t = current_data['low'].iloc[i-9:i+1].min()
            
            medium_raw_return = close_t / close_t10 - 1
            medium_vol_adj_return = (close_t / close_t10 - 1) / (high_t9_t - low_t9_t) if (high_t9_t - low_t9_t) > 0 else 0
        else:
            medium_raw_return = 0
            medium_vol_adj_return = 0
        
        # Long-Term (20-day)
        if i >= 20:
            close_t20 = current_data['close'].iloc[i-20]
            high_t19_t = current_data['high'].iloc[i-19:i+1].max()
            low_t19_t = current_data['low'].iloc[i-19:i+1].min()
            
            long_raw_return = close_t / close_t20 - 1
            long_trend_strength = (close_t - low_t19_t) / (high_t19_t - low_t19_t) if (high_t19_t - low_t19_t) > 0 else 0
        else:
            long_raw_return = 0
            long_trend_strength = 0
        
        # Volume Momentum Analysis
        volume_t = current_data['volume'].iloc[i]
        
        # Volume Rate of Change
        if i >= 3:
            volume_t3 = current_data['volume'].iloc[i-3]
            volume_roc_short = volume_t / volume_t3 - 1
        else:
            volume_roc_short = 0
            
        if i >= 10:
            volume_t10 = current_data['volume'].iloc[i-10]
            volume_roc_medium = volume_t / volume_t10 - 1
        else:
            volume_roc_medium = 0
            
        if i >= 20:
            volume_t20 = current_data['volume'].iloc[i-20]
            volume_roc_long = volume_t / volume_t20 - 1
        else:
            volume_roc_long = 0
        
        # Divergence Classification
        price_momentum_indicators = [short_raw_return, short_range_adj_return, medium_raw_return, 
                                   medium_vol_adj_return, long_raw_return, long_trend_strength]
        volume_roc_indicators = [volume_roc_short, volume_roc_medium, volume_roc_long]
        
        positive_price_count = sum(1 for x in price_momentum_indicators if x > 0)
        negative_price_count = sum(1 for x in price_momentum_indicators if x < 0)
        positive_volume_count = sum(1 for x in volume_roc_indicators if x > 0.15)
        negative_volume_count = sum(1 for x in volume_roc_indicators if x < -0.15)
        positive_volume_weak_count = sum(1 for x in volume_roc_indicators if x > 0)
        negative_volume_weak_count = sum(1 for x in volume_roc_indicators if x < 0)
        
        # Assign divergence scores
        if positive_price_count == len(price_momentum_indicators) and positive_volume_count == len(volume_roc_indicators):
            divergence_score = 2.0  # Strong Bullish
        elif positive_price_count > len(price_momentum_indicators)//2 and positive_volume_weak_count > len(volume_roc_indicators)//2:
            divergence_score = 1.0  # Weak Bullish
        elif negative_price_count == len(price_momentum_indicators) and negative_volume_count == len(volume_roc_indicators):
            divergence_score = -2.0  # Strong Bearish
        elif negative_price_count > len(price_momentum_indicators)//2 and negative_volume_weak_count > len(volume_roc_indicators)//2:
            divergence_score = -1.0  # Weak Bearish
        else:
            divergence_score = 0.0  # Neutral
        
        # Market Regime Detection
        # Volatility-Based Regimes
        if i >= 10:
            high_10d = current_data['high'].iloc[i-9:i+1].max()
            low_10d = current_data['low'].iloc[i-9:i+1].min()
            range_10d = (high_10d - low_10d) / close_t
            
            high_20d = current_data['high'].iloc[i-19:i+1].max()
            low_20d = current_data['low'].iloc[i-19:i+1].min()
            range_20d = (high_20d - low_20d) / close_t
            
            if range_10d < 0.6 * range_20d:
                volatility_regime = 'low'
            elif range_10d > 1.4 * range_20d:
                volatility_regime = 'high'
            else:
                volatility_regime = 'normal'
        else:
            volatility_regime = 'normal'
        
        # Trend-Based Regimes
        if i >= 20:
            trend_direction = np.sign(close_t - current_data['close'].iloc[i-20])
            trend_strength = (close_t - current_data['close'].iloc[i-10]) / (high_t9_t - low_t9_t) if (high_t9_t - low_t9_t) > 0 else 0
            
            if trend_direction > 0 and trend_strength > 0.3:
                trend_regime = 'strong_uptrend'
            elif trend_direction > 0 and trend_strength <= 0.3:
                trend_regime = 'weak_uptrend'
            elif trend_direction < 0 and trend_strength >= -0.3:
                trend_regime = 'weak_downtrend'
            elif trend_direction < 0 and trend_strength < -0.3:
                trend_regime = 'strong_downtrend'
            else:
                trend_regime = 'sideways'
        else:
            trend_regime = 'sideways'
        
        # Dynamic Factor Combination
        # Volatility Regime Weights
        if volatility_regime == 'low':
            short_weight, medium_weight, long_weight = 0.6, 0.3, 0.1
        elif volatility_regime == 'normal':
            short_weight, medium_weight, long_weight = 0.4, 0.4, 0.2
        else:  # high volatility
            short_weight, medium_weight, long_weight = 0.2, 0.3, 0.5
        
        # Calculate weighted momentum
        weighted_momentum = (short_weight * (short_raw_return + short_range_adj_return) / 2 +
                           medium_weight * (medium_raw_return + medium_vol_adj_return) / 2 +
                           long_weight * (long_raw_return + long_trend_strength) / 2)
        
        # Trend Regime Multipliers
        volume_confirmation = np.mean([volume_roc_short, volume_roc_medium, volume_roc_long])
        
        if trend_regime == 'strong_uptrend':
            trend_multiplier = 1.0 if volume_confirmation > 0.1 else 0.5
        elif trend_regime == 'weak_uptrend':
            trend_multiplier = 1.0 if volume_confirmation > 0.05 else 0.7
        elif trend_regime == 'sideways':
            trend_multiplier = 1.0 if volume_confirmation > 0.2 else 0.3
        elif trend_regime == 'weak_downtrend':
            trend_multiplier = 1.0 if volume_confirmation < -0.05 else 0.7
        elif trend_regime == 'strong_downtrend':
            trend_multiplier = 1.0 if volume_confirmation < -0.1 else 0.5
        else:
            trend_multiplier = 1.0
        
        # Volume Outlier Enhancement
        if i >= 20:
            volume_20d_avg = current_data['volume'].iloc[i-19:i+1].mean()
            volume_10d_avg = current_data['volume'].iloc[i-9:i+1].mean()
            
            # Relative Volume Positioning
            current_vs_historical = volume_t / volume_20d_avg if volume_20d_avg > 0 else 1.0
            recent_vs_historical = volume_10d_avg / volume_20d_avg if volume_20d_avg > 0 else 1.0
            
            # Volume Acceleration
            if i >= 2:
                volume_acceleration = (volume_t / current_data['volume'].iloc[i-1]) / (current_data['volume'].iloc[i-1] / current_data['volume'].iloc[i-2]) if current_data['volume'].iloc[i-1] > 0 and current_data['volume'].iloc[i-2] > 0 else 1.0
            else:
                volume_acceleration = 1.0
            
            # Outlier Classification
            if current_vs_historical > 4.0:
                volume_multiplier = 2.5
            elif current_vs_historical > 2.0:
                volume_multiplier = 1.8
            elif current_vs_historical > 0.7:
                volume_multiplier = 1.0
            elif current_vs_historical > 0.3:
                volume_multiplier = 0.4
            else:
                volume_multiplier = 0.1
            
            # Volume Trend Adjustments
            if volume_acceleration > 1.1:
                volume_trend_adjustment = 0.3
            elif volume_acceleration < 0.9:
                volume_trend_adjustment = -0.2
            else:
                volume_trend_adjustment = 0.0
        else:
            volume_multiplier = 1.0
            volume_trend_adjustment = 0.0
        
        # Final Factor Calculation
        base_factor = divergence_score * weighted_momentum * trend_multiplier
        enhanced_factor = base_factor * volume_multiplier * (1 + volume_trend_adjustment)
        
        factor_values.iloc[i] = enhanced_factor
    
    # Fill NaN values with 0
    factor_values = factor_values.fillna(0)
    
    return factor_values
