import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Volume-Confirmed Volatility Breakout Factor
    # Calculate 20-day high
    high_20d = df['high'].rolling(window=20).max()
    
    # Calculate breakout magnitude
    breakout_magnitude = (df['close'] - high_20d.shift(1)) / high_20d.shift(1)
    
    # Calculate 20-day volatility (std of daily returns)
    daily_returns = df['close'].pct_change()
    volatility_20d = daily_returns.rolling(window=20).std()
    
    # Calculate 60-day median volatility
    median_volatility_60d = volatility_20d.rolling(window=60).median()
    
    # Volume surge calculation
    volume_20d_avg = df['volume'].rolling(window=20).mean()
    volume_ratio = df['volume'] / volume_20d_avg
    
    # Volatility regime classification
    high_vol_regime = volatility_20d > median_volatility_60d
    low_vol_regime = ~high_vol_regime
    
    # Combined factor construction
    breakout_factor = pd.Series(0.0, index=df.index)
    
    # High volatility regime conditions
    high_vol_condition = (high_vol_regime & 
                         (volume_ratio > 1.5) & 
                         (breakout_magnitude > 0.02))
    breakout_factor[high_vol_condition] = breakout_magnitude[high_vol_condition]
    
    # Low volatility regime conditions
    low_vol_condition = (low_vol_regime & 
                        (volume_ratio > 1.2) & 
                        (breakout_magnitude > 0.01))
    breakout_factor[low_vol_condition] = breakout_magnitude[low_vol_condition]
    
    # Intraday Strength Persistence Factor
    # Calculate intraday return and efficiency
    intraday_return = (df['close'] - df['open']) / df['open']
    intraday_efficiency = (df['close'] - df['open']) / (df['high'] - df['low'])
    intraday_efficiency_abs = intraday_efficiency.abs()
    
    # Detect persistence pattern
    streak_length = pd.Series(0, index=df.index)
    strength_accumulation = pd.Series(0.0, index=df.index)
    avg_efficiency = pd.Series(0.0, index=df.index)
    
    for i in range(1, len(df)):
        if (intraday_return.iloc[i] * intraday_return.iloc[i-1]) > 0:
            streak_length.iloc[i] = streak_length.iloc[i-1] + 1
            strength_accumulation.iloc[i] = strength_accumulation.iloc[i-1] + intraday_return.iloc[i]
            avg_efficiency.iloc[i] = (avg_efficiency.iloc[i-1] * (streak_length.iloc[i]-1) + 
                                    intraday_efficiency_abs.iloc[i]) / streak_length.iloc[i]
        else:
            streak_length.iloc[i] = 1
            strength_accumulation.iloc[i] = intraday_return.iloc[i]
            avg_efficiency.iloc[i] = intraday_efficiency_abs.iloc[i]
    
    # Volume context
    volume_5d_avg = df['volume'].rolling(window=5).mean()
    volume_ratio_5d = df['volume'] / volume_5d_avg
    
    # Final persistence factor
    persistence_factor = (streak_length * strength_accumulation * avg_efficiency * 
                         volume_ratio_5d * np.sign(intraday_return))
    
    # Price-Volume Divergence Reversal Factor
    # Calculate 5-day trends
    price_trend_5d = df['close'] / df['close'].shift(5) - 1
    volume_trend_5d = df['volume'] / df['volume'].shift(5) - 1
    
    # Identify divergence patterns
    bullish_divergence = (price_trend_5d < 0) & (volume_trend_5d > 0)
    bearish_divergence = (price_trend_5d > 0) & (volume_trend_5d < 0)
    
    # Calculate 10-day volatility
    volatility_10d = daily_returns.rolling(window=10).std()
    
    # Divergence factor
    divergence_factor = pd.Series(0.0, index=df.index)
    divergence_factor[bullish_divergence] = (price_trend_5d.abs() * volume_trend_5d.abs() * 
                                           volatility_10d)[bullish_divergence]
    divergence_factor[bearish_divergence] = -(price_trend_5d.abs() * volume_trend_5d.abs() * 
                                            volatility_10d)[bearish_divergence]
    
    # Opening Gap Mean Reversion Factor
    # Calculate gap percentage
    gap_percentage = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Historical gap context
    abs_gap_20d_avg = gap_percentage.abs().rolling(window=20).mean()
    
    # Volume confirmation
    gap_volume_ratio = df['volume'] / volume_20d_avg
    
    # Mean reversion signal
    gap_factor = pd.Series(0.0, index=df.index)
    
    # Large gap scenarios
    large_gap = gap_percentage.abs() > (2 * abs_gap_20d_avg)
    gap_factor[large_gap & (gap_percentage < 0)] = gap_percentage[large_gap & (gap_percentage < 0)] * gap_volume_ratio[large_gap & (gap_percentage < 0)]
    gap_factor[large_gap & (gap_percentage > 0)] = gap_percentage[large_gap & (gap_percentage > 0)] * gap_volume_ratio[large_gap & (gap_percentage > 0)]
    
    # Small gap scenarios (weak signal)
    small_gap = gap_percentage.abs() < (0.5 * abs_gap_20d_avg)
    gap_factor[small_gap] = gap_percentage[small_gap] * gap_volume_ratio[small_gap] * 0.3
    
    # Multi-Timeframe Momentum Alignment Factor
    # Calculate momentum across timeframes
    momentum_3d = df['close'] / df['close'].shift(3) - 1
    momentum_10d = df['close'] / df['close'].shift(10) - 1
    momentum_20d = df['close'] / df['close'].shift(20) - 1
    
    # Assess alignment quality
    aligned_3_10 = (momentum_3d * momentum_10d) > 0
    aligned_3_20 = (momentum_3d * momentum_20d) > 0
    aligned_10_20 = (momentum_10d * momentum_20d) > 0
    
    full_alignment = aligned_3_10 & aligned_3_20 & aligned_10_20
    partial_alignment = (aligned_3_10.astype(int) + aligned_3_20.astype(int) + aligned_10_20.astype(int)) >= 2
    
    # Volume multiplier
    volume_multiplier = df['volume'] / volume_20d_avg
    
    # Final momentum factor
    momentum_factor = pd.Series(0.0, index=df.index)
    momentum_factor[full_alignment] = (momentum_3d + momentum_10d + momentum_20d)[full_alignment] / 3 * volume_multiplier[full_alignment] * 1.5
    momentum_factor[partial_alignment & ~full_alignment] = (momentum_3d + momentum_10d + momentum_20d)[partial_alignment & ~full_alignment] / 3 * volume_multiplier[partial_alignment & ~full_alignment] * 1.0
    
    # Combine all factors with equal weights
    final_factor = (breakout_factor + persistence_factor + divergence_factor + 
                   gap_factor + momentum_factor) / 5
    
    return final_factor
