import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Momentum-Adjusted Volume Divergence
    def volume_trend_slope(volume_series, window=5):
        slopes = []
        for i in range(len(volume_series)):
            if i < window - 1:
                slopes.append(np.nan)
            else:
                y = volume_series.iloc[i-window+1:i+1].values
                x = np.arange(window)
                slope, _, _, _, _ = linregress(x, y)
                slopes.append(slope)
        return pd.Series(slopes, index=volume_series.index)
    
    volume_slope = volume_trend_slope(df['volume'])
    roc_10 = (df['close'] / df['close'].shift(10) - 1)
    momentum_adjusted_volume = volume_slope * roc_10
    
    # High-Low Range Efficiency
    true_range = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    ], axis=1).max(axis=1)
    price_change = abs(df['close'] - df['close'].shift(1))
    efficiency_ratio = price_change / true_range
    
    # Volatility-Regressed Volume Surge
    volume_ma = df['volume'].rolling(window=20).mean()
    volume_surge = df['volume'] / volume_ma
    high_low_volatility = (df['high'] - df['low']).rolling(window=20).std()
    volatility_regressed_volume = volume_surge / (1 + high_low_volatility)
    
    # Pressure-Based Reversal Indicator
    midpoint = (df['high'] + df['low']) / 2
    pressure_index = (df['close'] - midpoint) * df['volume']
    pressure_ma = pressure_index.rolling(window=5).mean()
    pressure_std = pressure_index.rolling(window=20).std()
    pressure_zscore = (pressure_index - pressure_ma) / pressure_std
    pressure_reversal = -pressure_zscore * df['volume']
    
    # Liquidity-Adjusted Momentum
    ret_5 = df['close'].pct_change(5)
    ret_10 = df['close'].pct_change(10)
    ret_20 = df['close'].pct_change(20)
    turnover_efficiency = df['amount'] / df['volume']
    liquidity_adjusted_momentum = (ret_5 + ret_10 + ret_20) * turnover_efficiency
    
    # Gap-Fill Probability
    overnight_gap = (df['open'] / df['close'].shift(1) - 1)
    gap_volatility_ratio = abs(overnight_gap) / (df['high'] - df['low']).rolling(window=10).mean()
    volume_ratio = df['volume'] / df['volume'].rolling(window=10).mean()
    gap_fill_probability = -abs(overnight_gap) * gap_volatility_ratio * volume_ratio
    
    # Volume-Weighted Price Levels
    def volume_weighted_levels(high_low_data, volume_data, window=10):
        levels = []
        for i in range(len(high_low_data)):
            if i < window - 1:
                levels.append(np.nan)
            else:
                window_high = high_low_data.iloc[i-window+1:i+1]
                window_volume = volume_data.iloc[i-window+1:i+1]
                high_volume = (window_high * window_volume).sum() / window_volume.sum()
                current_price = high_low_data.iloc[i]
                distance = abs(current_price - high_volume)
                levels.append(-distance * window_volume.iloc[-1])
        return pd.Series(levels, index=high_low_data.index)
    
    high_weighted = volume_weighted_levels(df['high'], df['volume'])
    low_weighted = volume_weighted_levels(df['low'], df['volume'])
    volume_weighted_levels_combined = (high_weighted + low_weighted) / 2
    
    # Momentum-Decay Rate
    def exponential_decay_fit(returns_series, window=10):
        decay_rates = []
        for i in range(len(returns_series)):
            if i < window - 1:
                decay_rates.append(np.nan)
            else:
                returns = returns_series.iloc[i-window+1:i+1].values
                try:
                    # Simple exponential decay approximation
                    log_returns = np.log1p(returns)
                    x = np.arange(window)
                    slope, _, _, _, _ = linregress(x, log_returns)
                    decay_rates.append(slope)
                except:
                    decay_rates.append(np.nan)
        return pd.Series(decay_rates, index=returns_series.index)
    
    rolling_returns = df['close'].pct_change().rolling(window=10).mean()
    decay_rate = exponential_decay_fit(rolling_returns)
    decay_acceleration = decay_rate.diff(5)
    
    # Volume-Volatility Correlation Break
    volume_volatility_corr = df['volume'].rolling(window=20).corr(df['high'] - df['low'])
    corr_change = volume_volatility_corr.diff(5)
    correlation_break = abs(corr_change) * df['volume']
    
    # Efficiency-Weighted Trend
    def trend_slope_noise_ratio(price_series, window=10):
        slopes = []
        noise_ratios = []
        for i in range(len(price_series)):
            if i < window - 1:
                slopes.append(np.nan)
                noise_ratios.append(np.nan)
            else:
                prices = price_series.iloc[i-window+1:i+1].values
                x = np.arange(window)
                slope, intercept, _, _, _ = linregress(x, prices)
                predicted = slope * x + intercept
                residuals = prices - predicted
                noise_ratio = np.std(residuals) / abs(slope) if abs(slope) > 0 else np.nan
                slopes.append(slope)
                noise_ratios.append(noise_ratio)
        return pd.Series(slopes, index=price_series.index), pd.Series(noise_ratios, index=price_series.index)
    
    trend_slope, noise_ratio = trend_slope_noise_ratio(df['close'])
    volume_impact = df['volume'] / df['volume'].rolling(window=20).mean()
    efficiency_weighted_trend = trend_slope / (noise_ratio * volume_impact)
    
    # Combine all factors with equal weights
    factors = [
        momentum_adjusted_volume,
        efficiency_ratio,
        volatility_regressed_volume,
        pressure_reversal,
        liquidity_adjusted_momentum,
        gap_fill_probability,
        volume_weighted_levels_combined,
        decay_acceleration,
        correlation_break,
        efficiency_weighted_trend
    ]
    
    # Normalize each factor and combine
    combined_factor = pd.Series(0, index=df.index)
    for factor in factors:
        normalized_factor = (factor - factor.rolling(window=20).mean()) / factor.rolling(window=20).std()
        combined_factor = combined_factor + normalized_factor.fillna(0)
    
    return combined_factor
