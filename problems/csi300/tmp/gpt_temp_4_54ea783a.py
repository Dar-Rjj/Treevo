import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate a novel alpha factor combining multiple technical indicators:
    - Volatility-adjusted momentum
    - Volume-price divergence
    - Cumulative breakout strength
    - Range efficiency trend
    - Amount-driven momentum
    - Support-resistance bounce
    - Multi-timeframe momentum convergence
    """
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(20, len(df)):
        # Multi-Day Volatility-Adjusted Momentum
        if i >= 10:
            cum_return = (df['close'].iloc[i] - df['close'].iloc[i-10]) / df['close'].iloc[i-10]
            
            # Calculate 10-day ATR
            tr_values = []
            for j in range(i-9, i+1):
                high_low = df['high'].iloc[j] - df['low'].iloc[j]
                high_close_prev = abs(df['high'].iloc[j] - df['close'].iloc[j-1])
                low_close_prev = abs(df['low'].iloc[j] - df['close'].iloc[j-1])
                tr = max(high_low, high_close_prev, low_close_prev)
                tr_values.append(tr)
            atr_10 = np.mean(tr_values)
            volatility_momentum = cum_return / atr_10 if atr_10 != 0 else 0
        else:
            volatility_momentum = 0
        
        # Volume-Weighted Price Acceleration
        if i >= 4:
            vwap_5 = sum(df['close'].iloc[i-j] * df['volume'].iloc[i-j] for j in range(5)) / sum(df['volume'].iloc[i-j] for j in range(5))
            sma_5 = np.mean([df['close'].iloc[i-j] for j in range(5)])
            volume_divergence = vwap_5 - sma_5
        else:
            volume_divergence = 0
        
        # Cumulative Breakout Strength
        if i >= 19:
            high_20 = max(df['high'].iloc[i-j] for j in range(20))
            cumulative_breakout = sum((df['close'].iloc[i-j] - high_20) / high_20 for j in range(5))
            avg_volume_5 = np.mean([df['volume'].iloc[i-j] for j in range(5)])
            breakout_strength = cumulative_breakout * avg_volume_5
        else:
            breakout_strength = 0
        
        # Range Efficiency Trend
        if i >= 9:
            current_efficiency = np.mean([abs(df['close'].iloc[i-j] - df['open'].iloc[i-j]) / (df['high'].iloc[i-j] - df['low'].iloc[i-j]) for j in range(5)])
            prev_efficiency = np.mean([abs(df['close'].iloc[i-j-5] - df['open'].iloc[i-j-5]) / (df['high'].iloc[i-j-5] - df['low'].iloc[i-j-5]) for j in range(5)])
            efficiency_change = current_efficiency - prev_efficiency
            efficiency_momentum = efficiency_change * current_efficiency
        else:
            efficiency_momentum = 0
        
        # Amount-Driven Price Momentum
        if i >= 5:
            cumulative_amount = sum(df['amount'].iloc[i-j] for j in range(5))
            price_change = df['close'].iloc[i] - df['close'].iloc[i-5]
            amount_momentum = price_change / cumulative_amount if cumulative_amount != 0 else 0
        else:
            amount_momentum = 0
        
        # Support-Resistance Bounce Factor
        if i >= 19:
            low_20 = min(df['low'].iloc[i-j] for j in range(20))
            high_20 = max(df['high'].iloc[i-j] for j in range(20))
            bounce_ratio = (df['close'].iloc[i] - low_20) / (high_20 - low_20) if (high_20 - low_20) != 0 else 0
            bounce_factor = bounce_ratio * df['volume'].iloc[i]
        else:
            bounce_factor = 0
        
        # Multi-Timeframe Momentum Convergence
        if i >= 10:
            short_momentum = (df['close'].iloc[i] - df['close'].iloc[i-3]) / df['close'].iloc[i-3] if df['close'].iloc[i-3] != 0 else 0
            medium_momentum = (df['close'].iloc[i] - df['close'].iloc[i-10]) / df['close'].iloc[i-10] if df['close'].iloc[i-10] != 0 else 0
            momentum_convergence = short_momentum * medium_momentum
        else:
            momentum_convergence = 0
        
        # Combine all factors with equal weighting
        combined_factor = (
            volatility_momentum + 
            volume_divergence + 
            breakout_strength + 
            efficiency_momentum + 
            amount_momentum + 
            bounce_factor + 
            momentum_convergence
        ) / 7
        
        result.iloc[i] = combined_factor
    
    return result
