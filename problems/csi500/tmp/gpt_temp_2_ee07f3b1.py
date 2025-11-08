import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # High-Low Range Persistence
    # Calculate True Range
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    amount = df['amount']
    
    # True Range calculation
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Rolling averages
    atr_5 = true_range.rolling(window=5).mean()
    atr_20 = true_range.rolling(window=20).mean()
    
    # Persistence signal
    range_persistence = atr_5 / atr_20
    
    # Volume-Weighted Price Acceleration
    # Price momentum
    mom_3 = close.pct_change(periods=3)
    mom_6 = close.pct_change(periods=6)
    
    # Volume changes
    vol_3 = volume.pct_change(periods=3)
    vol_6 = volume.pct_change(periods=6)
    
    # Acceleration factor
    price_acceleration = mom_3 / mom_6
    vol_ratio = (1 + vol_3) / (1 + vol_6)
    acceleration_factor = (price_acceleration * vol_ratio) ** 2
    
    # Close Position Relative to Daily Range
    # Daily range position
    daily_range = high - low
    range_position = (close - low) / daily_range
    
    # Historical percentile
    hist_positions = range_position.rolling(window=21).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0
    )
    
    # Reversion signal
    reversion_signal = hist_positions * np.sign(range_position - 0.5)
    
    # Volume-Intensity Adjusted Volatility
    # Price volatility
    returns = close.pct_change()
    volatility = returns.rolling(window=10).std()
    
    # Volume intensity
    vol_avg_20 = volume.rolling(window=20).mean()
    volume_intensity = np.log(volume / vol_avg_20)
    
    # Combined signal
    vol_adj_volatility = np.cbrt(volatility * volume_intensity) * np.sign(returns)
    
    # Asymmetric Volume-Price Relationship
    # Separate up and down days
    returns_daily = close.pct_change()
    up_days = returns_daily > 0
    down_days = returns_daily < 0
    
    # Volume ratios
    up_vol = volume[up_days].rolling(window=20).mean()
    down_vol = volume[down_days].rolling(window=20).mean()
    
    # Asymmetry factor
    vol_ratio_asym = up_vol / down_vol
    # Inverse hyperbolic sine transformation
    vol_asymmetry = np.arcsinh(vol_ratio_asym) * np.sign(returns_daily.rolling(window=5).mean())
    
    # Gap-Fill Probability Indicator
    # Price gaps
    overnight_gap = (df['open'] - close.shift(1)) / close.shift(1)
    
    # Historical fill rates
    def calculate_fill_rate(gap_series):
        if len(gap_series) < 63:
            return 0.5
        filled = ((gap_series > 0) & (gap_series.shift(-1) < gap_series)) | \
                ((gap_series < 0) & (gap_series.shift(-1) > gap_series))
        return filled.mean()
    
    fill_probability = overnight_gap.rolling(window=63).apply(calculate_fill_rate, raw=False)
    
    # Predictive signal
    gap_signal = overnight_gap * fill_probability * np.sign(-overnight_gap)
    
    # Intraday Momentum Persistence
    # Intraday strength
    intraday_strength = (close - df['open']) / (high - low)
    
    # Persistence count with exponential weighting
    def persistence_count(series):
        if len(series) < 2:
            return 0
        current_sign = np.sign(series.iloc[-1])
        count = 0
        weight = 1.0
        total_weight = 0
        weighted_count = 0
        
        for i in range(len(series)-2, -1, -1):
            if np.sign(series.iloc[i]) == current_sign:
                weighted_count += weight
                total_weight += weight
                weight *= 0.9  # Exponential decay
            else:
                break
        return weighted_count / total_weight if total_weight > 0 else 0
    
    persistence = intraday_strength.rolling(window=10).apply(persistence_count, raw=False)
    
    # Momentum factor
    momentum_factor = (persistence * intraday_strength) ** 3
    
    # Volume-Weighted Support/Resistance
    # Identify key price levels
    def find_key_levels(prices, window=20):
        highs = prices.rolling(window=window, center=True).max()
        lows = prices.rolling(window=window, center=True).min()
        return highs, lows
    
    resistance, support = find_key_levels(close)
    
    # Volume concentration
    def volume_at_level(price, levels, vol_data, threshold=0.02):
        proximity = abs(price - levels) / price
        near_level = proximity < threshold
        if near_level.any():
            return vol_data[near_level].mean()
        return vol_data.mean()
    
    vol_at_resistance = [volume_at_level(close.iloc[i], resistance, volume) for i in range(len(close))]
    vol_at_support = [volume_at_level(close.iloc[i], support, volume) for i in range(len(close))]
    
    vol_concentration_res = pd.Series(vol_at_resistance, index=close.index) / volume.rolling(window=20).mean()
    vol_concentration_sup = pd.Series(vol_at_support, index=close.index) / volume.rolling(window=20).mean()
    
    # Breakout signal with inverse distance weighting
    dist_to_res = (resistance - close) / close
    dist_to_sup = (close - support) / close
    
    breakout_signal = (vol_concentration_res / (dist_to_res + 0.01)) - (vol_concentration_sup / (dist_to_sup + 0.01))
    
    # Combine all factors with equal weighting
    factors = pd.DataFrame({
        'range_persistence': range_persistence,
        'acceleration': acceleration_factor,
        'reversion': reversion_signal,
        'vol_adj_vol': vol_adj_volatility,
        'vol_asymmetry': vol_asymmetry,
        'gap_signal': gap_signal,
        'momentum': momentum_factor,
        'breakout': breakout_signal
    })
    
    # Standardize and combine
    factors_standardized = factors.apply(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x)
    combined_factor = factors_standardized.mean(axis=1)
    
    return combined_factor
