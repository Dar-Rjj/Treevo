import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Dynamic Volatility-Adjusted Price Momentum
    # Compute rolling volatility using high and low prices over past 20 days
    daily_range = (data['high'] - data['low']).rolling(window=20).std()
    
    # Calculate price momentum (close at t vs t-10)
    price_momentum = data['close'] - data['close'].shift(10)
    
    # Calculate 5-day volume moving average
    volume_ma_5 = data['volume'].rolling(window=5).mean()
    
    # Adjust momentum by volatility and scale by volume trend
    volatility_adjusted_momentum = price_momentum / (daily_range + 1e-8) * volume_ma_5
    
    # Relative Strength Mean Reversion
    # Since we don't have sector index, use market average as proxy
    market_avg = data['close'].rolling(window=50).mean()
    relative_strength = data['close'] / market_avg
    rs_ma_5 = relative_strength.rolling(window=5).mean()
    rs_ma_20 = relative_strength.rolling(window=20).mean()
    deviation = rs_ma_5 - rs_ma_20
    
    # Calculate 5-day volume change percentage
    volume_change_pct = data['volume'].pct_change(periods=5)
    daily_price_range = data['high'] - data['low']
    
    # Apply volume confirmation and price range scaling
    volume_confirmed_deviation = deviation * np.abs(volume_change_pct) / (daily_price_range + 1e-8)
    
    # Intraday Pressure Accumulation
    # Calculate buy-sell pressure
    intraday_pressure = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # Accumulate pressure over past 5 days with exponential weighting
    weights = np.array([0.5, 0.25, 0.125, 0.0625, 0.0625])  # Exponential weights
    accumulated_pressure = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 4:
            window_data = intraday_pressure.iloc[i-4:i+1].values
            accumulated_pressure.iloc[i] = np.sum(weights * window_data)
        else:
            accumulated_pressure.iloc[i] = 0
    
    # Calculate volume percentile over past 20 days
    volume_rank = data['volume'].rolling(window=20).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-8), 
        raw=False
    )
    
    # Combine pressure with volume profile
    pressure_volume_combined = accumulated_pressure * volume_rank
    
    # Liquidity-Adjusted Reversal Signal
    # Calculate 3-day return and identify extreme moves
    ret_3d = data['close'].pct_change(periods=3)
    ret_std_20 = ret_3d.rolling(window=20).std()
    extreme_moves = ret_3d / (ret_std_20 + 1e-8)
    
    # Compute Amihud illiquidity ratio (absolute return / volume)
    daily_ret = data['close'].pct_change()
    amihud_ratio = (np.abs(daily_ret) / (data['volume'] + 1e-8)).rolling(window=10).mean()
    
    # Calculate bid-ask spread proxy
    spread_proxy = ((data['high'] - data['low']) / data['close']).rolling(window=5).mean()
    
    # Combine liquidity measures
    liquidity_measure = amihud_ratio * spread_proxy
    
    # Calculate volume acceleration (5-day volume change rate)
    volume_acceleration = data['volume'].pct_change(periods=5)
    
    # Generate reversal signal
    reversal_signal = -extreme_moves * liquidity_measure * volume_acceleration
    
    # Efficiency Ratio Trend Strength
    # Calculate price efficiency (net change / total absolute changes)
    net_change_10d = data['close'] - data['close'].shift(10)
    total_abs_changes = np.abs(data['close'].diff()).rolling(window=10).sum()
    efficiency_ratio = net_change_10d / (total_abs_changes + 1e-8)
    
    # Calculate volume correlation with price moves (10-day)
    volume_correlation = data['close'].rolling(window=10).corr(data['volume'])
    
    # Combine for trend quality score
    trend_quality = efficiency_ratio * volume_correlation
    
    # Adjust for recent volatility regime
    current_volatility = daily_range
    avg_volatility_20d = daily_range.rolling(window=20).mean()
    volatility_ratio = current_volatility / (avg_volatility_20d + 1e-8)
    
    # Calculate close-to-high ratio for intraday strength
    close_to_high = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    
    # Final trend strength score
    trend_strength = trend_quality * volatility_ratio * close_to_high
    
    # Combine all factors with equal weighting
    factor = (
        volatility_adjusted_momentum.fillna(0) +
        volume_confirmed_deviation.fillna(0) +
        pressure_volume_combined.fillna(0) +
        reversal_signal.fillna(0) +
        trend_strength.fillna(0)
    )
    
    return factor
