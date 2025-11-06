import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Momentum with Volatility Adjustment
    short_vol = df['close'].rolling(window=5).std()
    medium_vol = df['close'].rolling(window=10).std()
    vol_adjusted_momentum = (df['close'] / df['close'].shift(5) - 1) / medium_vol
    
    momentum_persistence = (
        (df['close'] > df['close'].shift(1)).astype(int) +
        (df['close'].shift(1) > df['close'].shift(2)).astype(int) +
        (df['close'].shift(2) > df['close'].shift(3)).astype(int) +
        (df['close'].shift(3) > df['close'].shift(4)).astype(int) +
        (df['close'].shift(4) > df['close'].shift(5)).astype(int)
    ) / 5
    
    # Volume Confirmation Analysis
    volume_spike = df['volume'] / df['volume'].rolling(window=5).mean()
    volume_trend = df['volume'].rolling(window=3).mean() / df['volume'].rolling(window=6).mean() - 1
    
    # Calculate Price-Volume Correlation
    price_changes = df['close'].diff()
    volume_values = df['volume']
    price_volume_corr = pd.Series(index=df.index, dtype=float)
    for i in range(4, len(df)):
        if i >= 4:
            window_prices = price_changes.iloc[i-4:i+1]
            window_volumes = volume_values.iloc[i-4:i+1]
            if len(window_prices) >= 2 and not window_prices.isna().any() and not window_volumes.isna().any():
                corr_val = window_prices.corr(window_volumes)
                price_volume_corr.iloc[i] = corr_val if not np.isnan(corr_val) else 0
            else:
                price_volume_corr.iloc[i] = 0
    
    volume_confirmation = volume_spike * (1 + volume_trend) * (1 + price_volume_corr)
    
    # Intraday Price Efficiency
    capture_ratio = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    gap_efficiency = abs(df['open'] - df['close'].shift(1)) / (df['high'] - df['low']).replace(0, np.nan)
    efficiency_score = (capture_ratio + abs(df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)) / (1 + abs(gap_efficiency))
    
    # Amount-Based Quality Signals
    amount_momentum = df['amount'] / df['amount'].rolling(window=5).mean() - 1
    return_per_amount = (df['close'] / df['close'].shift(1) - 1) / df['amount'].replace(0, np.nan)
    quality_signal = amount_momentum * return_per_amount / (1 + df['amount'].rolling(window=5).std())
    
    # Multi-Timeframe Momentum Divergence
    very_short = df['close'] / df['close'].shift(1) - 1
    short = df['close'] / df['close'].shift(3) - 1
    medium = df['close'] / df['close'].shift(8) - 1
    divergence_score = np.sign(very_short) * np.sign(short) * np.sign(medium)
    
    # Alpha Factor Construction
    core_factor = vol_adjusted_momentum * momentum_persistence
    volume_adjusted = core_factor * volume_confirmation
    efficiency_enhanced = volume_adjusted * efficiency_score
    quality_filtered = efficiency_enhanced * (1 + quality_signal)
    
    # Apply divergence multiplier
    divergence_multiplier = np.where(divergence_score > 0, 1.5, 
                                   np.where(divergence_score < 0, 0.5, 1.0))
    
    final_alpha = quality_filtered * divergence_multiplier
    
    return final_alpha
