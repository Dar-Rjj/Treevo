import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate novel alpha factor combining multiple regime-adaptive signals,
    intraday patterns, multi-timeframe alignments, support/resistance dynamics,
    and price-volume correlation patterns.
    """
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate rolling windows
    for i in range(len(df)):
        if i < 20:  # Need at least 20 days for calculations
            result.iloc[i] = 0
            continue
            
        current_data = df.iloc[:i+1]
        
        # 1. Regime-Adaptive Momentum Signals
        # Volatility-Regime Momentum
        returns_5d = (current_data['close'].iloc[i] / current_data['close'].iloc[i-5]) - 1
        recent_returns = [(current_data['close'].iloc[j] / current_data['close'].iloc[j-1]) - 1 
                         for j in range(i-9, i+1)]
        vol_10d = np.std(recent_returns)
        median_vol = np.median([np.std([(current_data['close'].iloc[k] / current_data['close'].iloc[k-1]) - 1 
                                      for k in range(j-9, j+1)]) for j in range(10, i+1)])
        high_vol_regime = 1 if vol_10d > median_vol else 0
        vol_momentum = returns_5d * high_vol_regime
        
        # Volume-Regime Momentum Efficiency
        daily_return = (current_data['close'].iloc[i] / current_data['close'].iloc[i-1]) - 1
        median_volume = np.median(current_data['volume'].iloc[i-10:i+1])
        high_volume_regime = 1 if current_data['volume'].iloc[i] > median_volume else 0
        volume_momentum = daily_return * high_volume_regime
        
        # Trend Regime Persistence
        short_trend = current_data['close'].iloc[i] / current_data['close'].iloc[i-3]
        medium_trend = current_data['close'].iloc[i] / current_data['close'].iloc[i-10]
        trend_alignment = 1 if np.sign(short_trend - 1) == np.sign(medium_trend - 1) else 0
        trend_persistence = (short_trend - 1) * trend_alignment
        
        # 2. Intraday Pattern Strength
        # Gap Reversal Efficiency
        overnight_gap = (current_data['open'].iloc[i] - current_data['close'].iloc[i-1]) / current_data['close'].iloc[i-1]
        intraday_return = (current_data['close'].iloc[i] - current_data['open'].iloc[i]) / current_data['open'].iloc[i]
        gap_reversal = overnight_gap * intraday_return
        
        # Intraday Trend Consistency
        morning_momentum = (current_data['high'].iloc[i] - current_data['open'].iloc[i]) / current_data['open'].iloc[i]
        afternoon_momentum = (current_data['close'].iloc[i] - current_data['high'].iloc[i]) / current_data['high'].iloc[i]
        intraday_consistency = morning_momentum - afternoon_momentum
        
        # Range Breakout Efficiency
        daily_range = (current_data['high'].iloc[i] - current_data['low'].iloc[i]) / current_data['close'].iloc[i]
        avg_range_5d = np.mean([(current_data['high'].iloc[j] - current_data['low'].iloc[j]) / current_data['close'].iloc[j] 
                               for j in range(i-4, i+1)])
        range_breakout = daily_range / avg_range_5d if avg_range_5d != 0 else 0
        
        # 3. Multi-Timeframe Signal Alignment
        # Price-Volume Divergence Convergence
        short_divergence = (current_data['close'].iloc[i] / current_data['close'].iloc[i-3]) - (current_data['volume'].iloc[i] / current_data['volume'].iloc[i-3])
        medium_divergence = (current_data['close'].iloc[i] / current_data['close'].iloc[i-10]) - (current_data['volume'].iloc[i] / current_data['volume'].iloc[i-10])
        divergence_convergence = short_divergence * medium_divergence
        
        # Momentum Acceleration Alignment
        mom_3d = current_data['close'].iloc[i] / current_data['close'].iloc[i-3]
        mom_5d = current_data['close'].iloc[i] / current_data['close'].iloc[i-5]
        mom_10d = current_data['close'].iloc[i] / current_data['close'].iloc[i-10]
        momentum_alignment = (mom_3d / mom_5d) * (mom_5d / mom_10d) if mom_5d != 0 and mom_10d != 0 else 0
        
        # Volume Trend Confirmation
        short_vol_trend = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-3]
        medium_vol_trend = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-10]
        volume_confirmation = short_vol_trend / medium_vol_trend if medium_vol_trend != 0 else 0
        
        # 4. Support/Resistance Dynamics
        # Resistance Break Strength
        high_20d = max(current_data['high'].iloc[i-19:i+1])
        break_magnitude = current_data['close'].iloc[i] / high_20d
        resistance_break = break_magnitude * current_data['volume'].iloc[i]
        
        # Support Bounce Intensity
        low_20d = min(current_data['low'].iloc[i-19:i+1])
        bounce_magnitude = current_data['close'].iloc[i] / low_20d
        support_bounce = bounce_magnitude * current_data['amount'].iloc[i]
        
        # Range Breakout Confirmation
        range_breakout_ind = 1 if current_data['close'].iloc[i] > current_data['high'].iloc[i-1] else 0
        volume_confirmation_ind = 1 if current_data['volume'].iloc[i] > current_data['volume'].iloc[i-1] else 0
        breakout_confirmation = range_breakout_ind * volume_confirmation_ind
        
        # 5. Price-Volume Correlation Patterns
        # Direction Alignment Efficiency
        price_direction = np.sign(current_data['close'].iloc[i] - current_data['close'].iloc[i-1])
        volume_direction = np.sign(current_data['volume'].iloc[i] - current_data['volume'].iloc[i-1])
        direction_alignment = price_direction * volume_direction
        
        # Volume-Weighted Momentum
        price_change = current_data['close'].iloc[i] - current_data['close'].iloc[i-1]
        vol_weighted_change = price_change * current_data['volume'].iloc[i]
        vol_weighted_momentum = sum([(current_data['close'].iloc[j] - current_data['close'].iloc[j-1]) * current_data['volume'].iloc[j] 
                                   for j in range(i-2, i+1)])
        
        # Cumulative Divergence Intensity
        cum_price_change = (current_data['close'].iloc[i] / current_data['close'].iloc[i-5]) - 1
        cum_volume_change = (current_data['volume'].iloc[i] / current_data['volume'].iloc[i-5]) - 1
        divergence_intensity = abs(cum_price_change - cum_volume_change)
        
        # Combine all factors with equal weights
        factors = [
            vol_momentum, volume_momentum, trend_persistence,
            gap_reversal, intraday_consistency, range_breakout,
            divergence_convergence, momentum_alignment, volume_confirmation,
            resistance_break, support_bounce, breakout_confirmation,
            direction_alignment, vol_weighted_momentum, divergence_intensity
        ]
        
        # Normalize and combine
        result.iloc[i] = np.mean(factors)
    
    return result
