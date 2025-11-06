import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Volatility-Adjusted Price Momentum
    momentum = df['close'] / df['close'].shift(5) - 1
    vol = df['close'].rolling(window=20).std()
    vol_adj_momentum = momentum / (vol + 1e-8)
    
    # Volume-Price Divergence Factor
    price_trend = df['close'] / df['close'].rolling(window=5).mean()
    volume_trend = df['volume'] / df['volume'].rolling(window=5).mean()
    
    correlation = pd.Series(index=df.index, dtype=float)
    for i in range(10, len(df)):
        window_prices = price_trend.iloc[i-9:i+1]
        window_volumes = volume_trend.iloc[i-9:i+1]
        if window_prices.std() > 0 and window_volumes.std() > 0:
            correlation.iloc[i] = window_prices.corr(window_volumes)
        else:
            correlation.iloc[i] = 0
    
    divergence_factor = correlation * momentum
    
    # Intraday Pressure Indicator
    high_low_range = df['high'] - df['low']
    high_low_range = high_low_range.replace(0, np.nan)
    
    buying_pressure = ((df['close'] - df['low']) / high_low_range) * df['volume']
    selling_pressure = ((df['high'] - df['close']) / high_low_range) * df['volume']
    
    net_pressure = buying_pressure - selling_pressure
    smoothed_pressure = net_pressure.ewm(span=3).mean()
    
    # Regime-Sensitive Mean Reversion
    market_vol = df['close'].rolling(window=20).std()
    hist_vol_median = market_vol.rolling(window=252).median()
    
    price_deviation = df['close'] / df['close'].rolling(window=50).mean() - 1
    
    vol_regime = market_vol / hist_vol_median
    regime_adj_mean_rev = price_deviation / (vol_regime + 1e-8)
    
    # Liquidity-Weighted Return Momentum
    short_return = df['close'] / df['close'].shift(3) - 1
    avg_volume = df['volume'].rolling(window=20).mean()
    liquidity_ratio = df['volume'] / (avg_volume + 1e-8)
    
    liquidity_weighted_return = short_return * liquidity_ratio
    
    # Momentum persistence filter (3-day sign consistency)
    momentum_sign = np.sign(liquidity_weighted_return.rolling(window=3).sum())
    filtered_momentum = liquidity_weighted_return * momentum_sign
    
    # Combine all factors with equal weights
    combined_factor = (
        vol_adj_momentum.fillna(0) +
        divergence_factor.fillna(0) +
        smoothed_pressure.fillna(0) +
        regime_adj_mean_rev.fillna(0) +
        filtered_momentum.fillna(0)
    ) / 5
    
    return combined_factor
