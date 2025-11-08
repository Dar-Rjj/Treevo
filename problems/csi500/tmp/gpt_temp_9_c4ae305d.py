import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum Divergence Factor
    # Calculate rolling momentum
    momentum_10 = data['close'].pct_change(periods=10)
    momentum_20 = data['close'].pct_change(periods=20)
    
    # Calculate momentum divergence
    momentum_divergence = (momentum_10 - momentum_20).abs()
    
    # Volume confirmation
    volume_5d_avg = data['volume'].rolling(window=5).mean()
    volume_ratio = data['volume'] / volume_5d_avg
    momentum_factor = momentum_divergence * volume_ratio
    
    # High-Low Range Breakout Factor
    # Calculate True Range
    tr1 = data['high'] - data['low']
    tr2 = (data['high'] - data['close'].shift(1)).abs()
    tr3 = (data['low'] - data['close'].shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Compute range breakout signal
    range_5d_avg = (data['high'] - data['low']).rolling(window=5).mean()
    current_range = data['high'] - data['low']
    breakout_signal = (current_range > range_5d_avg).astype(int)
    
    # Volume-weighted breakout
    amount_10d_avg = data['amount'].rolling(window=10).mean()
    breakout_factor = breakout_signal * data['amount'] / amount_10d_avg
    
    # Volatility Regime Shift Factor
    # Calculate returns
    returns = data['close'].pct_change()
    
    # Calculate volatility measures
    vol_5d = returns.rolling(window=5).std()
    vol_20d = returns.rolling(window=20).std()
    
    # Detect regime shift
    vol_ratio = vol_5d / vol_20d
    # Bullish: volatility compression (ratio < 1 and decreasing)
    vol_ratio_change = vol_ratio.diff()
    bullish_signal = ((vol_ratio < 1) & (vol_ratio_change < 0)).astype(int)
    # Bearish: volatility expansion (ratio > 1 and increasing)
    bearish_signal = ((vol_ratio > 1) & (vol_ratio_change > 0)).astype(int)
    volatility_factor = bullish_signal - bearish_signal
    
    # Liquidity-Adjusted Price Pressure
    # Calculate price pressure
    close_open_gap = (data['close'] - data['open']) / data['open']
    high_low_range = (data['high'] - data['low']) / data['open']
    price_pressure = close_open_gap / (high_low_range + 1e-8)  # Avoid division by zero
    
    # Assess liquidity conditions
    volume_rank = data['volume'].rolling(window=20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    amount_rank = data['amount'].rolling(window=20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    liquidity_score = (volume_rank + amount_rank) / 2
    
    # Combine pressure and liquidity
    liquidity_factor = price_pressure * (1 - liquidity_score)  # Inverse relationship
    
    # Intraday Strength Persistence
    # Calculate intraday strength
    intraday_strength = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # Assess persistence pattern
    strength_sign = np.sign(intraday_strength)
    persistence_count = strength_sign.groupby((strength_sign != strength_sign.shift(1)).cumsum()).cumcount() + 1
    persistence_score = persistence_count * intraday_strength.abs()
    
    # Volume confirmation
    volume_trend = data['volume'].rolling(window=5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    intraday_factor = persistence_score * np.sign(volume_trend)
    
    # Abnormal Volume Return Factor
    # Detect abnormal volume
    volume_20d_avg = data['volume'].rolling(window=20).mean()
    volume_20d_std = data['volume'].rolling(window=20).std()
    volume_zscore = (data['volume'] - volume_20d_avg) / (volume_20d_std + 1e-8)
    
    # Calculate concurrent return
    returns_1d = data['close'].pct_change()
    
    # Generate predictive signal
    abnormal_volume_factor = volume_zscore * returns_1d
    
    # Price-Volume Convergence
    # Calculate trends using rolling regression
    def calc_slope(series, window=10):
        def linear_trend(x):
            if len(x) < 2:
                return np.nan
            return np.polyfit(range(len(x)), x, 1)[0]
        return series.rolling(window=window).apply(linear_trend, raw=False)
    
    price_trend = calc_slope(data['close'])
    volume_trend = calc_slope(data['volume'])
    
    # Detect convergence/divergence
    convergence_factor = price_trend * volume_trend
    
    # Range Expansion Momentum
    # Calculate daily range
    daily_range = (data['high'] - data['low']) / data['close'].shift(1)
    
    # Assess range expansion
    range_10d_avg = daily_range.rolling(window=10).mean()
    range_expansion = daily_range / range_10d_avg
    
    # Combine with price action
    close_strength = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    range_momentum = range_expansion * close_strength
    
    # Efficiency Ratio Acceleration
    # Calculate price efficiency
    def efficiency_ratio(price_series, period=10):
        net_change = price_series.diff(period).abs()
        total_movement = price_series.diff().abs().rolling(window=period).sum()
        return net_change / (total_movement + 1e-8)
    
    efficiency = efficiency_ratio(data['close'])
    efficiency_acceleration = efficiency.diff()
    
    # Volume confirmation
    volume_efficiency_corr = data['volume'].rolling(window=10).corr(efficiency)
    efficiency_factor = efficiency_acceleration * volume_efficiency_corr
    
    # Gap Fill Probability Factor
    # Calculate price gaps
    price_gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Assess gap fill tendency (simplified)
    gap_magnitude = price_gap.abs()
    # Historical tendency: larger gaps tend to fill more
    fill_probability = 1 / (1 + np.exp(-gap_magnitude * 10))  # Sigmoid function
    
    # Adjust for volume
    gap_volume_ratio = data['volume'] / data['volume'].rolling(window=20).mean()
    gap_factor = fill_probability * np.sign(-price_gap) * gap_volume_ratio
    
    # Combine all factors with equal weights
    factors = pd.DataFrame({
        'momentum': momentum_factor,
        'breakout': breakout_factor,
        'volatility': volatility_factor,
        'liquidity': liquidity_factor,
        'intraday': intraday_factor,
        'abnormal_volume': abnormal_volume_factor,
        'convergence': convergence_factor,
        'range_momentum': range_momentum,
        'efficiency': efficiency_factor,
        'gap': gap_factor
    })
    
    # Standardize each factor and combine
    combined_factor = factors.apply(lambda x: (x - x.mean()) / x.std()).mean(axis=1)
    
    return combined_factor
