import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Intraday Volatility-Adjusted Momentum
    # Momentum component
    momentum_3d = df['close'] / df['close'].shift(3) - 1
    intraday_range = (df['high'] - df['low']) / df['close']
    intraday_volatility = intraday_range.rolling(window=5).std()
    volatility_adjusted_momentum = momentum_3d / (intraday_volatility + 1e-8)
    
    # Volume-Price Divergence Factor
    # Price trend component
    price_change_5d = df['close'] / df['close'].shift(5) - 1
    volume_change_5d = df['volume'] / df['volume'].shift(5) - 1
    
    # Rolling correlation between price and volume changes
    rolling_corr = pd.Series(index=df.index, dtype=float)
    for i in range(20, len(df)):
        window_prices = price_change_5d.iloc[i-20:i]
        window_volumes = volume_change_5d.iloc[i-20:i]
        rolling_corr.iloc[i] = window_prices.corr(window_volumes)
    
    # Volume anomaly detection
    volume_mean = df['volume'].rolling(window=20).mean()
    volume_std = df['volume'].rolling(window=20).std()
    volume_zscore = (df['volume'] - volume_mean) / (volume_std + 1e-8)
    
    # Divergence: positive price change with negative volume z-score
    price_volume_divergence = (price_change_5d > 0) & (volume_zscore < 0)
    divergence_factor = price_change_5d * rolling_corr * price_volume_divergence.astype(float)
    
    # Acceleration-Deceleration Indicator
    # First derivative (velocity)
    velocity_short = df['close'] / df['close'].shift(2) - 1
    velocity_medium = df['close'] / df['close'].shift(5) - 1
    
    # Second derivative (acceleration)
    acceleration = velocity_short - velocity_medium
    normalized_acceleration = acceleration / (intraday_range + 1e-8)
    
    # Liquidity-Adjusted Reversal
    # Price reversal component
    recent_return = df['close'] / df['close'].shift(1) - 1
    historical_mean_return = df['close'].pct_change().rolling(window=20).mean()
    reversal_component = recent_return - historical_mean_return
    
    # Liquidity adjustment
    volume_volatility_ratio = df['volume'] / (df['high'] - df['low'] + 1e-8)
    liquidity_adjusted_reversal = reversal_component * volume_volatility_ratio
    
    # Trend Persistence Score
    # Multi-timeframe trend analysis
    def calculate_slope(series, window):
        x = np.arange(window)
        slopes = pd.Series(index=series.index, dtype=float)
        for i in range(window-1, len(series)):
            y = series.iloc[i-window+1:i+1].values
            if len(y) == window:
                slope = (window * np.sum(x*y) - np.sum(x) * np.sum(y)) / (window * np.sum(x**2) - np.sum(x)**2)
                slopes.iloc[i] = slope
        return slopes
    
    short_trend = calculate_slope(df['close'], 3)
    medium_trend = calculate_slope(df['close'], 10)
    long_trend = calculate_slope(df['close'], 20)
    
    # Consistency measurement
    trend_signs = pd.DataFrame({
        'short': np.sign(short_trend),
        'medium': np.sign(medium_trend),
        'long': np.sign(long_trend)
    })
    
    matches = (trend_signs['short'] == trend_signs['medium']).astype(int) + \
              (trend_signs['short'] == trend_signs['long']).astype(int) + \
              (trend_signs['medium'] == trend_signs['long']).astype(int)
    
    trend_strength = (abs(short_trend) + abs(medium_trend) + abs(long_trend)) / 3
    persistence_score = matches * trend_strength
    
    # Combine all factors with equal weights
    combined_factor = (
        volatility_adjusted_momentum.fillna(0) +
        divergence_factor.fillna(0) +
        normalized_acceleration.fillna(0) +
        liquidity_adjusted_reversal.fillna(0) +
        persistence_score.fillna(0)
    )
    
    return combined_factor
