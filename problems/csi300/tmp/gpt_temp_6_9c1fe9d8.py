import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Dynamic Volatility-Adjusted Price Momentum
    # Compute Rolling Price Momentum
    momentum_10d = df['close'].pct_change(10)
    momentum_20d = df['close'].pct_change(20)
    
    # Calculate Dynamic Volatility using Average True Range
    high_low = df['high'] - df['low']
    high_close_prev = abs(df['high'] - df['close'].shift(1))
    low_close_prev = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    atr_5d = true_range.rolling(5).mean()
    atr_10d = true_range.rolling(10).mean()
    
    # Combine Momentum with Volatility Adjustment
    adj_momentum_short = momentum_10d / atr_5d
    adj_momentum_long = momentum_20d / atr_10d
    volatility_adjusted_momentum = adj_momentum_short - adj_momentum_long
    
    # Volume-Expressed Price Reversal
    # Identify Recent Price Extremes
    highest_5d = df['close'].rolling(5).max()
    lowest_5d = df['close'].rolling(5).min()
    
    # Calculate Extreme-to-Current Distance
    dist_to_high = (highest_5d - df['close']) / df['close']
    dist_to_low = (df['close'] - lowest_5d) / df['close']
    
    # Weight by Volume Signature
    # Find volume on extreme days
    high_volume = df['volume'].where(df['close'] == highest_5d).rolling(5).max()
    low_volume = df['volume'].where(df['close'] == lowest_5d).rolling(5).max()
    current_volume = df['volume']
    
    volume_ratio_high = current_volume / high_volume.replace(0, np.nan)
    volume_ratio_low = current_volume / low_volume.replace(0, np.nan)
    
    volume_expressed_reversal = dist_to_high * volume_ratio_high - dist_to_low * volume_ratio_low
    
    # Intraday Pressure Accumulation
    # Calculate Daily Pressure
    intraday_range = (df['high'] - df['low']) / df['close']
    close_open_gap = (df['close'] - df['open']) / df['open']
    daily_pressure = intraday_range * close_open_gap
    
    # Accumulate with Volume Confirmation
    pressure_volume = daily_pressure * df['volume']
    cumulative_pressure_3d = pressure_volume.rolling(3).sum()
    cumulative_pressure_5d = pressure_volume.rolling(5).sum()
    pressure_accumulation = cumulative_pressure_3d - cumulative_pressure_5d
    
    # Liquidity-Adjusted Trend Strength
    # Determine Price Trend Direction
    ma_8d = df['close'].rolling(8).mean()
    ma_21d = df['close'].rolling(21).mean()
    trend_strength = (ma_8d - ma_21d) / df['close']
    
    # Assess Liquidity Conditions
    volume_trend = df['volume'].rolling(5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    amount_per_trade = df['amount'] / df['volume']
    amount_trend = amount_per_trade.rolling(5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    
    # Combine Trend and Liquidity
    liquidity_adjusted_trend = trend_strength * volume_trend * amount_trend
    
    # Opening Gap Momentum Persistence
    # Measure Opening Gaps
    prev_close = df['close'].shift(1)
    gap_size = abs(df['open'] - prev_close) / prev_close
    gap_direction = np.sign(df['open'] - prev_close)
    
    # Track Gap Filling Behavior
    gap_fill_high = (df['high'] - df['open']) * gap_direction / (gap_size * prev_close).replace(0, np.nan)
    gap_fill_low = (df['open'] - df['low']) * gap_direction / (gap_size * prev_close).replace(0, np.nan)
    gap_fill_pct = (gap_fill_high + gap_fill_low) / 2
    
    # Combine with Volume Dynamics
    opening_volume = df['volume'].rolling(3).mean()  # Use recent average as opening volume proxy
    prev_volume_trend = df['volume'].pct_change(3)
    gap_persistence = gap_fill_pct * opening_volume * prev_volume_trend
    
    # Multi-Timeframe Price-Volume Divergence
    # Short-term Analysis (1-3 days)
    price_change_3d = df['close'].pct_change(3)
    volume_change_3d = df['volume'].pct_change(3)
    short_term_divergence = price_change_3d - volume_change_3d
    
    # Medium-term Analysis (5-10 days)
    price_volume_corr = df['close'].rolling(10).corr(df['volume'])
    volume_accel = df['volume'].pct_change(5) - df['volume'].pct_change(10)
    medium_term_divergence = price_volume_corr * volume_accel
    
    # Long-term Context (20+ days)
    volume_structural = df['volume'].rolling(20).std() / df['volume'].rolling(20).mean()
    price_volume_persistence = df['close'].rolling(20).corr(df['volume'])
    long_term_divergence = volume_structural * price_volume_persistence
    
    multi_timeframe_divergence = short_term_divergence + medium_term_divergence + long_term_divergence
    
    # Volatility Regime Adaptive Factor
    # Classify Volatility Environment
    vol_20d = true_range.rolling(20).mean()
    high_vol_period = vol_20d > vol_20d.rolling(50).quantile(0.7)
    low_vol_period = vol_20d < vol_20d.rolling(50).quantile(0.3)
    
    # Calculate Regime-Specific Signals
    momentum_5d = df['close'].pct_change(5)
    mean_reversion_5d = -df['close'].pct_change(5)  # Negative for mean reversion
    
    regime_signal = pd.Series(0, index=df.index)
    regime_signal[high_vol_period] = momentum_5d[high_vol_period]
    regime_signal[low_vol_period] = mean_reversion_5d[low_vol_period]
    
    # Combine with Volume Confirmation
    volume_in_regime = df['volume'].pct_change(5)
    regime_adaptive_factor = regime_signal * volume_in_regime
    
    # Combine all factors with equal weights
    factor = (volatility_adjusted_momentum + 
              volume_expressed_reversal + 
              pressure_accumulation + 
              liquidity_adjusted_trend + 
              gap_persistence + 
              multi_timeframe_divergence + 
              regime_adaptive_factor) / 7
    
    return factor
