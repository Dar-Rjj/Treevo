import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility-Adjusted Price Momentum Divergence
    # Compute Short-Term Price Momentum (5-day return)
    short_term_momentum = data['close'] / data['close'].shift(5) - 1
    
    # Compute Medium-Term Price Momentum (20-day return)
    medium_term_momentum = data['close'] / data['close'].shift(20) - 1
    
    # Calculate Momentum Divergence
    momentum_divergence = short_term_momentum - medium_term_momentum
    
    # Adjust by Recent Volatility (20-day std of returns)
    returns = data['close'].pct_change()
    volatility = returns.rolling(window=20).std()
    volatility_adjusted_momentum = momentum_divergence / (volatility + 1e-8)
    
    # Volume-Driven Reversal with Price Pressure
    # Identify High Volume Events
    avg_volume_20 = data['volume'].rolling(window=20).mean()
    high_volume_flag = data['volume'] > (2 * avg_volume_20)
    
    # Compute Price Reversal Signal
    daily_return = data['close'].pct_change()
    reversal_signal = -daily_return * high_volume_flag
    
    # Incorporate Price Pressure
    price_pressure = (data['high'] - data['low']) / data['close']
    reversal_with_pressure = reversal_signal * price_pressure
    
    # Aggregate Signal Strength over past 5 high volume days
    high_volume_days = high_volume_flag.astype(int)
    volume_weight = data['volume'] / avg_volume_20
    aggregated_reversal = reversal_with_pressure.rolling(window=5).sum() * volume_weight
    
    # Intraday Strength Persistence Factor
    # Calculate Intraday Strength Ratio
    intraday_strength = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    
    # Detect Persistence Pattern (consecutive days with ratio > 0.7)
    strong_close = intraday_strength > 0.7
    persistence_streak = strong_close.astype(int) * (strong_close.groupby((~strong_close).cumsum()).cumcount() + 1)
    
    # Adjust for Market Regime (using 5-day return as proxy)
    stock_5d_return = data['close'] / data['close'].shift(5) - 1
    # Since we don't have index data, use cross-sectional normalization
    market_adjusted = stock_5d_return / (stock_5d_return.rolling(window=20).std() + 1e-8)
    
    # Incorporate Volume Confirmation
    volume_ratio = data['volume'] / avg_volume_20
    persistence_factor = persistence_streak * market_adjusted * volume_ratio
    
    # Liquidity-Efficient Trend Following
    # Compute Efficiency-Adjusted Trend
    price_change_10d = data['close'] / data['close'].shift(10) - 1
    dollar_volume = data['close'] * data['volume']
    avg_dollar_volume = dollar_volume.rolling(window=10).mean()
    efficiency_trend = price_change_10d / (avg_dollar_volume + 1e-8)
    
    # Identify Low-Liquidity Acceleration
    volume_rank = data['volume'].rolling(window=20).rank(pct=True)
    low_liquidity_multiplier = 1 - volume_rank
    liquidity_adjusted_trend = efficiency_trend * low_liquidity_multiplier
    
    # Incorporate Breakout Confirmation
    rolling_20d_high = data['high'].rolling(window=20).max()
    breakout_premium = (data['close'] > rolling_20d_high.shift(1)).astype(float)
    breakout_trend = liquidity_adjusted_trend + (breakout_premium * 0.1)
    
    # Smooth with Volatility Filter
    volatility_10d = returns.rolling(window=10).std()
    smoothed_trend = breakout_trend / (volatility_10d + 1e-8)
    
    # Mean-Reversion with Volume Clustering
    # Calculate Price Deviation from Mean
    ma_20 = data['close'].rolling(window=20).mean()
    price_deviation = (data['close'] - ma_20) / data['close']
    
    # Detect Volume Clustering Patterns (5-day autocorrelation)
    volume_autocorr = data['volume'].rolling(window=5).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
    ).fillna(0)
    
    # Combine Mean-Reversion with Volume
    mean_reversion_volume = price_deviation * volume_autocorr
    
    # Adjust for Recent Trend Strength
    trend_5d = data['close'] / data['close'].shift(5) - 1
    pure_mean_reversion = mean_reversion_volume - trend_5d
    
    # Combine all factors with equal weights
    final_factor = (
        volatility_adjusted_momentum.fillna(0) +
        aggregated_reversal.fillna(0) +
        persistence_factor.fillna(0) +
        smoothed_trend.fillna(0) +
        pure_mean_reversion.fillna(0)
    )
    
    return final_factor
