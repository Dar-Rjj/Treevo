import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining momentum-volume divergence, true range efficiency,
    volume-scaled extreme reversals, and regime-adaptive signals.
    """
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Required lookback periods
    required_lookback = 10
    
    for i in range(required_lookback, len(df)):
        current_data = df.iloc[i-required_lookback:i+1]
        
        # 1. Momentum-Volume Divergence
        # Price Momentum Components
        close_prices = current_data['close'].values
        short_momentum = close_prices[-1] / close_prices[-6] - 1
        medium_momentum = close_prices[-1] / close_prices[-11] - 1
        momentum_acceleration = (close_prices[-1] / close_prices[-6]) / (close_prices[-6] / close_prices[-11]) - 1
        
        # Volume Trend Components
        volumes = current_data['volume'].values
        volume_momentum = volumes[-1] / volumes[-6]
        volume_acceleration = (volumes[-1] / volumes[-6]) / (volumes[-6] / volumes[-11])
        
        # Volume persistence (last 5 days)
        vol_persistence = np.sum(volumes[-5:] > volumes[-6:-1]) / 5
        
        # Divergence Signals
        positive_divergence = short_momentum / volume_momentum if volume_momentum != 0 else 0
        negative_divergence = volume_momentum / short_momentum if short_momentum != 0 else 0
        acceleration_divergence = momentum_acceleration / volume_acceleration if volume_acceleration != 0 else 0
        
        momentum_volume_score = positive_divergence + negative_divergence + acceleration_divergence
        
        # 2. True Range Efficiency
        high_prices = current_data['high'].values
        low_prices = current_data['low'].values
        
        # True Range Calculations
        true_ranges = []
        for j in range(1, len(current_data)):
            tr1 = high_prices[j] - low_prices[j]
            tr2 = abs(high_prices[j] - close_prices[j-1])
            tr3 = abs(low_prices[j] - close_prices[j-1])
            true_ranges.append(max(tr1, tr2, tr3))
        
        daily_efficiency = abs(close_prices[-1] - close_prices[-2]) / true_ranges[-1] if true_ranges[-1] != 0 else 0
        
        # 3-day and 5-day cumulative ranges
        cum_range_3d = sum(true_ranges[-3:]) if len(true_ranges) >= 3 else 0
        cum_range_5d = sum(true_ranges[-5:]) if len(true_ranges) >= 5 else 0
        
        efficiency_3d = abs(close_prices[-1] - close_prices[-4]) / cum_range_3d if cum_range_3d != 0 else 0
        efficiency_trend = efficiency_3d / daily_efficiency if daily_efficiency != 0 else 0
        
        # Efficiency Patterns
        high_eff_persistence = sum(eff > 0.7 for eff in [abs(close_prices[k] - close_prices[k-1]) / true_ranges[k-1] 
                                                         for k in range(-5, 0)]) / 5
        
        efficiency_score = daily_efficiency + efficiency_3d + efficiency_trend + high_eff_persistence
        
        # 3. Volume-Scaled Extreme Reversals
        # Extreme Move Detection
        price_deviation_3d = (close_prices[-1] - np.mean(close_prices[-4:])) / np.std(close_prices[-4:]) if np.std(close_prices[-4:]) != 0 else 0
        
        range_position = (close_prices[-1] - np.min(low_prices[-6:])) / (np.max(high_prices[-6:]) - np.min(low_prices[-6:])) if (np.max(high_prices[-6:]) - np.min(low_prices[-6:])) != 0 else 0.5
        
        gap_extreme = abs(current_data['open'].iloc[-1] / close_prices[-2] - 1)
        
        # Volume Confirmation
        volume_zscore = (volumes[-1] - np.mean(volumes[-6:])) / np.std(volumes[-6:]) if np.std(volumes[-6:]) != 0 else 0
        volume_spike = volumes[-1] / np.median(volumes[-11:]) if np.median(volumes[-11:]) != 0 else 1
        
        # Reversal Signals
        deviation_reversal = price_deviation_3d * volume_zscore * np.sign(close_prices[-1] - close_prices[-2])
        range_reversal = range_position * volume_spike * (1 - 2 * range_position)
        gap_reversal = gap_extreme * volume_spike * np.sign(close_prices[-1] - current_data['open'].iloc[-1])
        
        reversal_score = deviation_reversal + range_reversal + gap_reversal
        
        # 4. Regime-Adaptive Signals
        # Volatility Regime
        short_vol = np.std(close_prices[-6:]) if len(close_prices) >= 6 else 0
        medium_vol = np.std(close_prices[-11:]) if len(close_prices) >= 11 else 0
        volatility_regime = short_vol / medium_vol if medium_vol != 0 else 1
        
        # Volume Regime
        volume_regime = volumes[-1] / np.mean(volumes[-11:]) if np.mean(volumes[-11:]) != 0 else 1
        
        # Adaptive Signals
        high_vol_signal = volatility_regime * volume_regime * short_momentum
        low_vol_signal = (1 / volatility_regime) * volume_spike * deviation_reversal if volatility_regime != 0 else 0
        transition_signal = abs(volatility_regime - 1) * vol_persistence * efficiency_trend
        
        regime_score = high_vol_signal + low_vol_signal + transition_signal
        
        # Combine all components with equal weighting
        final_score = (momentum_volume_score + efficiency_score + reversal_score + regime_score) / 4
        
        result.iloc[i] = final_score
    
    # Forward fill any NaN values
    result = result.fillna(method='ffill')
    
    return result
