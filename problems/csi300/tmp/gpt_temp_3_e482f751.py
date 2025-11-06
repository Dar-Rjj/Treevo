import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Required columns
    cols = ['open', 'high', 'low', 'close', 'amount', 'volume']
    
    for i in range(len(df)):
        if i < 20:  # Minimum window for calculations
            result.iloc[i] = 0
            continue
            
        current_data = df.iloc[:i+1]  # Only current and past data
        
        # 1. Momentum Acceleration
        # Time-decayed multi-period returns with exponential weights
        weights = np.exp(-0.1 * np.arange(10))[::-1]  # Recent weights higher
        returns = current_data['close'].pct_change().iloc[-10:].values
        
        if len(returns) == 10:
            weighted_momentum = np.sum(weights * returns)
            momentum_accel = weighted_momentum - np.sum(weights[:5] * returns[:5]) / np.sum(weights[:5])
        else:
            momentum_accel = 0
        
        # 2. Volatility-Scaled Reversal
        rolling_mean = current_data['close'].rolling(window=10).mean().iloc[-1]
        price_deviation = (current_data['close'].iloc[-1] - rolling_mean) / rolling_mean
        
        # Current volatility (10-day rolling std)
        current_vol = current_data['close'].pct_change().rolling(window=10).std().iloc[-1]
        if current_vol > 0:
            vol_scaled_reversal = price_deviation / current_vol
        else:
            vol_scaled_reversal = 0
        
        # 3. Volume-Validated Breakout
        # Rolling channel (10-day high/low)
        high_10 = current_data['high'].rolling(window=10).max().iloc[-1]
        low_10 = current_data['low'].rolling(window=10).min().iloc[-1]
        
        # Breakout detection
        current_close = current_data['close'].iloc[-1]
        channel_position = (current_close - low_10) / (high_10 - low_10) if (high_10 - low_10) > 0 else 0.5
        
        # Volume relative to 10-day moving average
        volume_ma = current_data['volume'].rolling(window=10).mean().iloc[-1]
        current_volume = current_data['volume'].iloc[-1]
        volume_ratio = current_volume / volume_ma if volume_ma > 0 else 1
        
        volume_breakout = (channel_position - 0.5) * volume_ratio
        
        # 4. Cumulative Order Flow
        # Volume-weighted daily pressure (close vs open)
        daily_pressure = ((current_data['close'] - current_data['open']) / current_data['open']) * current_data['volume']
        
        # Decaying sum (exponential decay with half-life of 5 days)
        decay_weights = np.exp(-0.1386 * np.arange(len(daily_pressure)))  # 0.1386 ≈ ln(2)/5
        cumulative_flow = np.sum(decay_weights[::-1] * daily_pressure.values) / np.sum(decay_weights)
        
        # Normalize by average volume
        avg_volume = current_data['volume'].mean()
        if avg_volume > 0:
            cumulative_flow /= avg_volume
        
        # 5. Regime-Adaptive Weighting
        # Market condition classification based on volatility regime
        vol_regime = current_vol / current_data['close'].pct_change().std() if len(current_data) > 20 else 1
        
        # Dynamic factor emphasis based on regime
        if vol_regime > 1.2:  # High volatility regime
            weights = [0.2, 0.4, 0.2, 0.2]  # Emphasize reversal
        elif vol_regime < 0.8:  # Low volatility regime
            weights = [0.4, 0.1, 0.3, 0.2]  # Emphasize momentum
        else:  # Normal regime
            weights = [0.3, 0.3, 0.2, 0.2]
        
        # Combine factors with regime-adaptive weights
        combined_factor = (weights[0] * momentum_accel + 
                          weights[1] * vol_scaled_reversal + 
                          weights[2] * volume_breakout + 
                          weights[3] * cumulative_flow)
        
        result.iloc[i] = combined_factor
    
    return result
