import pandas as pd
import pandas as pd

def heuristics_v2(df, lookback_period=20, n_days=5, m_days=10, sma_period=50):
    # Volume-Weighted High-Low Spread
    df['high_low_vol'] = (df['high'] - df['low']) * df['volume']
    
    # Summarize Over Lookback Period
    total_high_low_vol = df['high_low_vol'].rolling(window=lookback_period).sum()
    total_volume = df['volume'].rolling(window=lookback_period).sum()
    
    # Calculate Momentum
    average_high_low_vol = total_high_low_vol / total_volume
    current_high_low_vol = df['high_low_vol']
    momentum = current_high_low_vol - average_high_low_vol
    
    # Incorporate Price Trend
    price_change = df['close'] - df['open']
    momentum += price_change
    
    # Intraday Range Growth
    today_range = df['high'] - df['low']
    prev_day_range = (df['high'].shift(1) - df['low'].shift(1)).fillna(0)
    intraday_range_growth = (today_range - prev_day_range) / (prev_day_range + 1e-6)
    
    # Volume Weighted Moving Average
    close_prices = df['close'].rolling(window=lookback_period).apply(lambda x: (x * df.loc[x.index, 'volume']).sum() / df.loc[x.index, 'volume'].sum(), raw=False)
    
    # Combine Intraday Range Growth and Volume Weighted Moving Average
    combined_factor = intraday_range_growth + close_prices
    
    # Calculate Price Momentum
    close_momentum = df['close'].pct_change(n_days)
    open_momentum = df['open'].pct_change(n_days)
    price_momentum = (close_momentum + open_momentum) / 2
    
    # Adjust for Volume Surge
    n_day_volume_change = df['volume'] - df['volume'].shift(n_days)
    m_day_volume_change = df['volume'] - df['volume'].shift(m_days)
    volume_surge = n_day_volume_change + m_day_volume_change
    volume_surge[volume_surge < 0] *= 0.8
    
    # Final Combined Alpha Factor
    alpha_factor = (price_momentum * momentum) + intraday_range_growth + volume_surge + close_prices
    
    # 50-day Simple Moving Average (SMA) of closing prices
    sma_50 = df['close'].rolling(window=sma_period).mean()
    relative_position = (df['close'] > sma_50).astype(int) - (df['close'] < sma_50).astype(int)
    
    # Momentum in High and Low Prices
    high_momentum = df['high'] - df['high'].shift(1)
    low_momentum = df['low'] - df['low'].shift(1)
    combined_momentum = high_momentum + low_momentum
    momentum_signal = (combined_momentum > 0).astype(int) - (combined_momentum < 0).astype(int)
    
    # Analyze Volume Patterns
    volume_change_pct = df['volume'].pct_change().fillna(0)
    
    # Leverage Price and Volume for an Intensity Factor
    intensity_factor = (df['close'].pct_change() * df['volume']).rolling(window=20).mean()
    
    # Combine all factors
    final_alpha_factor = alpha_factor + relative_position + momentum_signal + intensity_factor
    
    return final_alpha_factor
