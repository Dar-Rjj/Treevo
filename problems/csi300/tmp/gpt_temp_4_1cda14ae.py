import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Momentum with Volume Confirmation
    # Calculate 5-day price ROC
    price_roc = df['close'].pct_change(periods=5)
    
    # Calculate volume trend slope (5-day linear regression slope)
    volume_trend = df['volume'].rolling(window=5).apply(
        lambda x: np.polyfit(range(5), x, 1)[0] if not x.isnull().any() else np.nan, 
        raw=False
    )
    
    # 20-day average volume for scaling
    avg_volume_20 = df['volume'].rolling(window=20).mean()
    
    # Combine momentum and volume confirmation
    momentum_volume = price_roc * volume_trend / avg_volume_20
    
    # Volatility-Adjusted Intraday Strength
    # Intraday price strength
    intraday_range = (df['high'] - df['low']) / df['open']
    close_position = (df['close'] - df['low']) / (df['high'] - df['low'])
    intraday_strength = intraday_range * close_position
    
    # Historical volatility (20-day std of returns)
    returns = df['close'].pct_change()
    volatility_20 = returns.rolling(window=20).std()
    
    # Volatility-adjusted intraday strength
    vol_adjusted_strength = intraday_strength / volatility_20
    
    # Volume-Price Divergence Factor
    # Price trend comparison
    ma_10 = df['close'].rolling(window=10).mean()
    ma_20 = df['close'].rolling(window=20).mean()
    price_trend = (ma_10 - ma_20) / ma_20
    
    # Volume anomaly (Z-score)
    volume_mean_20 = df['volume'].rolling(window=20).mean()
    volume_std_20 = df['volume'].rolling(window=20).std()
    volume_zscore = (df['volume'] - volume_mean_20) / volume_std_20
    
    # Volume-price divergence
    volume_price_divergence = price_trend * volume_zscore
    
    # Acceleration-Deceleration Indicator
    # Price acceleration
    ret_5 = df['close'].pct_change(periods=5)
    ret_10 = df['close'].pct_change(periods=10)
    acceleration = ret_5 - ret_10
    
    # Volume intensity ratio
    volume_5_avg = df['volume'].rolling(window=5).mean()
    volume_20_avg = df['volume'].rolling(window=20).mean()
    volume_ratio = volume_5_avg / volume_20_avg
    
    # Weighted acceleration
    weighted_acceleration = acceleration * volume_ratio
    
    # Relative Strength with Volume Support
    # Since we don't have market data, use price momentum as proxy
    stock_return_10 = df['close'].pct_change(periods=10)
    # Using price momentum as relative strength proxy
    relative_strength = stock_return_10
    
    # Volume percentile (20-day rolling)
    volume_percentile = df['volume'].rolling(window=20).apply(
        lambda x: (x[-1] > x[:-1]).mean() if len(x) == 20 else np.nan,
        raw=True
    )
    
    # Volume-supported relative strength
    volume_supported_rs = relative_strength * volume_percentile
    
    # Gap Persistence Factor
    # Opening gap size
    prev_close = df['close'].shift(1)
    gap_size = abs((df['open'] - prev_close) / prev_close)
    
    # Intraday recovery metric
    intraday_recovery = (df['close'] - df['open']) / (df['high'] - df['low'])
    # Adjust sign based on gap direction
    gap_direction = np.sign(df['open'] - prev_close)
    signed_recovery = intraday_recovery * gap_direction
    
    # Gap persistence
    gap_persistence = gap_size * signed_recovery
    gap_persistence_avg = gap_persistence.rolling(window=5).mean()
    
    # Volume-Weighted Price Efficiency
    # Price efficiency ratio
    price_change_abs = abs(df['close'] - df['open'])
    trading_range = df['high'] - df['low']
    efficiency_ratio = price_change_abs / trading_range
    
    # Volume concentration (simplified - using volume relative to range)
    volume_concentration = df['volume'] / trading_range
    
    # Volume-weighted efficiency
    volume_weighted_efficiency = efficiency_ratio * volume_concentration
    
    # Combine all factors with equal weights
    factors = pd.DataFrame({
        'momentum_volume': momentum_volume,
        'vol_adjusted_strength': vol_adjusted_strength,
        'volume_price_divergence': volume_price_divergence,
        'weighted_acceleration': weighted_acceleration,
        'volume_supported_rs': volume_supported_rs,
        'gap_persistence': gap_persistence_avg,
        'volume_weighted_efficiency': volume_weighted_efficiency
    })
    
    # Standardize each factor and take simple average
    factor_values = factors.apply(lambda x: (x - x.mean()) / x.std()).mean(axis=1)
    
    return factor_values
