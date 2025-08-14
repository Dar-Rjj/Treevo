import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Volume-Weighted Average Price (VWAP)
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Daily VWAP Change
    df['vwap_change'] = df['vwap'] - df['vwap'].shift(1)
    
    # Separate Positive and Negative Changes
    positive_changes = df['vwap_change'].where(df['vwap_change'] > 0, 0)
    negative_changes = df['vwap_change'].where(df['vwap_change'] < 0, 0).abs()
    
    # 14-Day Averages for RSI
    avg_positive_14 = positive_changes.rolling(window=14).mean()
    avg_negative_14 = negative_changes.rolling(window=14).mean()
    
    # Calculate RSI
    relative_strength = avg_positive_14 / avg_negative_14
    rsi = 100 - (100 / (1 + relative_strength))
    
    # Intraday High-Low Spread
    high_low_spread = df['high'] - df['low']
    
    # Volume-Adjusted High-Low Spread
    volume_adjusted_high_low_spread = high_low_spread * df['volume']
    
    # High-Close to Open-High Ratio
    high_close_diff = df['high'] - df['close']
    open_high_diff = df['open'] - df['high']
    high_close_ratio = high_close_diff / open_high_diff.replace(0, float('inf'))
    
    # Daily Returns
    daily_returns = df['close'].pct_change()
    
    # Long-Term Volume-Weighted Return (Momentum Component)
    long_term_vw_return = (daily_returns.rolling(window=90) * df['volume']).sum() / df['volume'].rolling(window=90).sum()
    
    # Short-Term Volume-Weighted Return (Reversal Component)
    short_term_vw_return = (daily_returns.rolling(window=5) * df['volume']).sum() / df['volume'].rolling(window=5).sum()
    
    # Adjusted VW-Return
    adjusted_vw_return = long_term_vw_return - short_term_vw_return
    
    # Momentum Component
    current_high_low_volume = (df['high'] - df['low']) * df['volume']
    past_n_days_avg_high_low_volume = (current_high_low_volume.rolling(window=90) * df['volume']).sum() / df['volume'].rolling(window=90).sum()
    momentum_component = current_high_low_volume - past_n_days_avg_high_low_volume
    
    # Price-Volume Imbalance Factor
    price_volume_imbalance = (df['close'] - df['open']) * df['volume']
    price_volume_imbalance_factor = price_volume_imbalance.rolling(window=90).sum()
    
    # Combine Volume-Adjusted High-Low Spread, High-Close to Open-High Ratio, and Price-Volume Imbalance
    combined_factor = (volume_adjusted_high_low_spread + high_close_ratio + price_volume_imbalance_factor)
    
    # Synthesize Final Alpha Factor
    alpha_factor = (rsi + high_low_spread + adjusted_vw_return + momentum_component + combined_factor)
    
    return alpha_factor
