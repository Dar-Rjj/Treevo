import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining multiple heuristics:
    1. Volatility-Adjusted Volume Trend Divergence
    2. Intraday Momentum Persistence Factor
    3. Relative Turnover Efficiency Factor
    4. Amplitude-Volume Correlation Breakout
    5. Liquidity-Adjusted Reversal Strength
    """
    
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    alpha_factor = pd.Series(index=data.index, dtype=float)
    
    # 1. Volatility-Adjusted Volume Trend Divergence
    # Calculate Rolling Volume Trend
    vol_ma_short = data['volume'].rolling(window=5, min_periods=3).mean()
    vol_ma_long = data['volume'].rolling(window=20, min_periods=10).mean()
    vol_trend_ratio = (vol_ma_short / vol_ma_long) - 1
    
    # Calculate Price Volatility
    high_low_range = data['high'] - data['low']
    price_volatility = (high_low_range / data['close']) * 100
    
    # Combine Volume Trend with Volatility
    vol_adj_trend = vol_trend_ratio * price_volatility.abs()
    vol_adj_trend = np.where(vol_trend_ratio >= 0, 
                            vol_adj_trend.abs(), 
                            -vol_adj_trend.abs())
    
    # 2. Intraday Momentum Persistence Factor
    # Calculate Consecutive Same-Direction Moves
    daily_direction = np.where(data['close'] > data['open'], 1, -1)
    
    # Count consecutive same signs
    streak_length = pd.Series(index=data.index, dtype=int)
    current_streak = 1
    current_direction = daily_direction[0]
    
    for i in range(len(data)):
        if i == 0:
            streak_length.iloc[i] = 1
        else:
            if daily_direction[i] == current_direction:
                current_streak += 1
            else:
                current_streak = 1
                current_direction = daily_direction[i]
            streak_length.iloc[i] = current_streak
    
    # Measure Move Strength
    abs_return = abs(data['close'] - data['open']) / data['open']
    momentum_strength = abs_return * streak_length * daily_direction
    
    # Adjust for Volume Confirmation
    vol_median = data['volume'].rolling(window=20, min_periods=10).median()
    vol_ratio = data['volume'] / vol_median
    intraday_momentum = momentum_strength * vol_ratio
    
    # 3. Relative Turnover Efficiency Factor
    # Calculate Price Efficiency Ratio
    price_change = (data['high'] - data['low']) / data['close']
    trading_turnover = (data['volume'] * data['close']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # For sector comparison, we'll use cross-sectional ranking
    efficiency_ratio = trading_turnover.rolling(window=5, min_periods=3).mean()
    efficiency_rank = efficiency_ratio.rolling(window=60, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) >= 20 else np.nan
    )
    
    # Apply Momentum Filter
    price_trend = data['close'].pct_change(5)
    trend_direction = np.sign(price_trend)
    turnover_efficiency = efficiency_rank * trend_direction
    
    # 4. Amplitude-Volume Correlation Breakout
    # Calculate Price Amplitude
    daily_amplitude = ((data['high'] - data['low']) / data['close']) * 100
    amplitude_vol = daily_amplitude.rolling(window=20, min_periods=10).std()
    
    # Measure Volume-Amplitude Relationship
    correlation_window = 20
    corr_values = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if i >= correlation_window - 1:
            window_data = data.iloc[i-correlation_window+1:i+1]
            corr = window_data['volume'].corr(window_data['high'] - window_data['low'])
            corr_values.iloc[i] = corr if not np.isnan(corr) else 0
        else:
            corr_values.iloc[i] = 0
    
    # Detect Correlation Changes
    corr_mean = corr_values.rolling(window=60, min_periods=20).mean()
    corr_std = corr_values.rolling(window=60, min_periods=20).std().replace(0, 1)
    corr_zscore = (corr_values - corr_mean) / corr_std
    
    # Identify Breakout Signals
    range_midpoint = (data['high'] + data['low']) / 2
    range_direction = np.where(data['close'] > range_midpoint, 1, -1)
    
    vol_median_20 = data['volume'].rolling(window=20, min_periods=10).median()
    breakout_signal = np.where(
        (abs(corr_zscore) > 2) & (data['volume'] > vol_median_20),
        corr_zscore * range_direction,
        0
    )
    
    # 5. Liquidity-Adjusted Reversal Strength
    # Calculate Short-Term Reversal
    prev_return = data['close'].pct_change(1)
    reversal_signal = -prev_return
    
    # Assess Liquidity Conditions
    vol_to_vol_ratio = data['volume'] / (data['high'] - data['low']).replace(0, np.nan)
    
    # Rolling percentile rank for liquidity
    liquidity_rank = vol_to_vol_ratio.rolling(window=60, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) >= 20 else 0.5
    )
    
    # Combine Reversal with Liquidity
    reversal_strength = reversal_signal * liquidity_rank
    reversal_strength = np.sign(reversal_strength) * np.sqrt(abs(reversal_strength))
    
    # Combine all factors with equal weights
    factors = pd.DataFrame({
        'vol_trend': vol_adj_trend,
        'intraday_momentum': intraday_momentum,
        'turnover_eff': turnover_efficiency,
        'breakout': breakout_signal,
        'reversal': reversal_strength
    })
    
    # Normalize each factor by z-score
    for col in factors.columns:
        mean_val = factors[col].rolling(window=60, min_periods=20).mean()
        std_val = factors[col].rolling(window=60, min_periods=20).std().replace(0, 1)
        factors[col] = (factors[col] - mean_val) / std_val
    
    # Final alpha factor (equal weighted combination)
    alpha_factor = factors.mean(axis=1)
    
    return alpha_factor
