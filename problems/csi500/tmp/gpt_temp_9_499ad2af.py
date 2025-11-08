import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate novel alpha factor combining multiple market signals:
    - Volatility-adjusted momentum
    - Volume-weighted intraday pressure  
    - Multi-timeframe trend alignment
    - Intraday momentum persistence
    - Liquidity-adjusted return impact
    """
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(max(20, len(df))):
        if i < 20:
            result.iloc[i] = 0
            continue
            
        # Volatility-Adjusted Price Momentum
        if i >= 5:
            raw_momentum = df['close'].iloc[i] / df['close'].iloc[i-5] - 1
            daily_range = df['high'].iloc[i] - df['low'].iloc[i]
            volatility_adj_momentum = raw_momentum / daily_range if daily_range > 0 else 0
        else:
            volatility_adj_momentum = 0
        
        # Volume-Weighted Intraday Pressure
        price_pressure = (df['close'].iloc[i] - (df['high'].iloc[i] + df['low'].iloc[i])/2) / ((df['high'].iloc[i] - df['low'].iloc[i])/2) if (df['high'].iloc[i] - df['low'].iloc[i]) > 0 else 0
        volume_avg_20 = df['volume'].iloc[i-19:i+1].mean()
        volume_ratio = df['volume'].iloc[i] / volume_avg_20 if volume_avg_20 > 0 else 1
        volume_pressure = price_pressure * volume_ratio
        
        # 3-day cumulative sum
        if i >= 2:
            volume_pressure_cumsum = volume_pressure + (result.iloc[i-1] if i-1 >= 0 else 0) + (result.iloc[i-2] if i-2 >= 0 else 0)
        else:
            volume_pressure_cumsum = volume_pressure
        
        # Multi-Timeframe Trend Alignment
        if i >= 10:
            short_trend = df['close'].iloc[i] / df['close'].iloc[i-3] - 1
            medium_trend = df['close'].iloc[i] / df['close'].iloc[i-10] - 1
            trend_alignment = np.sign(short_trend * medium_trend)
        else:
            trend_alignment = 0
        
        # Intraday Momentum Persistence
        morning_signal = (df['high'].iloc[i] - df['open'].iloc[i]) / (df['open'].iloc[i] - df['low'].iloc[i]) if (df['open'].iloc[i] - df['low'].iloc[i]) > 0 else 0
        afternoon_signal = (df['close'].iloc[i] - df['low'].iloc[i]) / (df['high'].iloc[i] - df['close'].iloc[i]) if (df['high'].iloc[i] - df['close'].iloc[i]) > 0 else 0
        intraday_momentum = morning_signal * afternoon_signal
        
        # 5-day exponential weighting
        if i >= 4:
            alpha = 2 / (5 + 1)
            intraday_weighted = alpha * intraday_momentum + (1 - alpha) * (result.iloc[i-1] if i-1 >= 0 else 0)
        else:
            intraday_weighted = intraday_momentum
        
        # Liquidity-Adjusted Return Impact
        if i >= 1:
            raw_return_impact = abs(df['close'].iloc[i] / df['close'].iloc[i-1] - 1)
            liquidity_proxy = df['volume'].iloc[i] / (df['high'].iloc[i] - df['low'].iloc[i]) if (df['high'].iloc[i] - df['low'].iloc[i]) > 0 else df['volume'].iloc[i]
            liquidity_adj_return = raw_return_impact / liquidity_proxy if liquidity_proxy > 0 else 0
        else:
            liquidity_adj_return = 0
        
        # Combine all signals with equal weights
        combined_signal = (
            volatility_adj_momentum +
            volume_pressure_cumsum +
            trend_alignment +
            intraday_weighted +
            liquidity_adj_return
        ) / 5
        
        result.iloc[i] = combined_signal
    
    return result
