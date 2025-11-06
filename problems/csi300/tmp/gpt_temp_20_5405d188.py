import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Change Momentum with Volume Confirmation
    # Calculate Price Momentum
    short_term_momentum = (df['close'] / df['close'].shift(5) - 1)
    medium_term_momentum = (df['close'] / df['close'].shift(20) - 1)
    
    # Calculate Volume Trend
    short_term_volume_change = (df['volume'] / df['volume'].shift(5) - 1)
    
    # Volume Direction Consistency
    volume_increase_count = pd.Series(index=df.index, dtype=float)
    volume_decrease_count = pd.Series(index=df.index, dtype=float)
    
    for i in range(5, len(df)):
        window_volume = df['volume'].iloc[i-5:i+1]
        increases = (window_volume.diff() > 0).sum()
        decreases = (window_volume.diff() < 0).sum()
        volume_increase_count.iloc[i] = increases
        volume_decrease_count.iloc[i] = decreases
    
    volume_direction_score = (volume_increase_count - volume_decrease_count) / 5
    volume_trend_strength = (volume_increase_count + volume_decrease_count) / 5
    
    # Combine Momentum with Volume Confirmation
    weighted_short_term = short_term_momentum * volume_direction_score
    weighted_medium_term = medium_term_momentum * volume_trend_strength
    momentum_signal = (weighted_short_term + weighted_medium_term) / 2
    
    # High-Low Range Breakout Efficiency
    # Calculate True Range Pattern
    daily_range_pct = (df['high'] - df['low']) / df['low']
    range_5d_avg = daily_range_pct.rolling(window=5, min_periods=3).mean()
    range_20d_avg = daily_range_pct.rolling(window=20, min_periods=10).mean()
    range_expansion_ratio = range_5d_avg / range_20d_avg
    
    # Assess Breakout Quality
    close_position = (df['close'] - df['low']) / (df['high'] - df['low'])
    
    # Breakout Persistence
    upper_quartile_count = pd.Series(index=df.index, dtype=float)
    expanding_range_count = pd.Series(index=df.index, dtype=float)
    
    for i in range(1, len(df)):
        # Count consecutive days closing in upper range quartile
        j = i
        while j >= 0 and close_position.iloc[j] > 0.75:
            j -= 1
        upper_quartile_count.iloc[i] = i - j
        
        # Count consecutive days with expanding range
        j = i
        while j > 0 and daily_range_pct.iloc[j] > daily_range_pct.iloc[j-1]:
            j -= 1
        expanding_range_count.iloc[i] = i - j
    
    breakout_quality = (upper_quartile_count * expanding_range_count) / 25
    
    # Generate Range Efficiency Signal
    range_efficiency = range_expansion_ratio * breakout_quality
    
    # Volatility-Regressed Price Reversal
    # Calculate Recent Volatility Regime
    returns = df['close'].pct_change()
    rolling_volatility = returns.rolling(window=20, min_periods=10).std()
    vol_median = rolling_volatility.rolling(window=60, min_periods=30).median()
    
    high_vol_regime = (rolling_volatility > vol_median).astype(float)
    low_vol_regime = (rolling_volatility <= vol_median).astype(float)
    
    # Measure Price Reversal Strength
    ma_20 = df['close'].rolling(window=20, min_periods=10).mean()
    deviation = (df['close'] - ma_20) / ma_20
    normalized_reversal = deviation / (rolling_volatility + 1e-8)
    
    # Reversal Momentum
    reversal_acceleration = normalized_reversal.diff()
    
    # Generate Volatility-Adjusted Reversal Factor
    vol_weighted_reversal = (normalized_reversal * 1.5 * high_vol_regime + 
                            normalized_reversal * 0.5 * low_vol_regime)
    
    # Liquidity-Efficient Price Movement
    # Calculate Price Movement per Unit Volume
    price_change = df['close'].diff().abs()
    min_volume = df['volume'].rolling(window=20).quantile(0.1)
    volume_safe = df['volume'].clip(lower=min_volume)
    price_per_volume = price_change / volume_safe
    
    # Assess Liquidity Efficiency Trend
    efficiency_5ma = price_per_volume.rolling(window=5, min_periods=3).mean()
    efficiency_momentum = price_per_volume / efficiency_5ma - 1
    
    # Generate Liquidity Signal
    price_direction = np.sign(df['close'].diff())
    liquidity_signal = efficiency_momentum * price_direction
    
    # Opening Gap Filling Probability
    # Analyze Gap Patterns
    opening_gap = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Calculate Gap Filling Metrics
    gap_filled_pct = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if opening_gap.iloc[i] > 0:  # Gap up
            if df['low'].iloc[i] <= df['close'].shift(1).iloc[i]:
                gap_filled_pct.iloc[i] = 1.0
            else:
                min_fill = (df['low'].iloc[i] - df['close'].shift(1).iloc[i]) / opening_gap.iloc[i]
                gap_filled_pct.iloc[i] = max(0, 1 - min_fill)
        elif opening_gap.iloc[i] < 0:  # Gap down
            if df['high'].iloc[i] >= df['close'].shift(1).iloc[i]:
                gap_filled_pct.iloc[i] = 1.0
            else:
                min_fill = (df['close'].shift(1).iloc[i] - df['high'].iloc[i]) / abs(opening_gap.iloc[i])
                gap_filled_pct.iloc[i] = max(0, 1 - min_fill)
        else:
            gap_filled_pct.iloc[i] = 0
    
    # Historical filling probability (simplified)
    hist_fill_prob = gap_filled_pct.rolling(window=60, min_periods=20).mean()
    
    # Generate Gap Trading Signal
    gap_signal = opening_gap * hist_fill_prob
    
    # Combine all signals with equal weights
    combined_signal = (
        momentum_signal.fillna(0) * 0.2 +
        range_efficiency.fillna(0) * 0.2 +
        vol_weighted_reversal.fillna(0) * 0.2 +
        liquidity_signal.fillna(0) * 0.2 +
        gap_signal.fillna(0) * 0.2
    )
    
    return combined_signal
