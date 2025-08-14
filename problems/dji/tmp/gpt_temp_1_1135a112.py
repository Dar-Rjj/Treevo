import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Range
    intraday_range = df['high'] - df['low']
    
    # Calculate Momentum
    n = 5  # Lookback period for momentum
    momentum = df['close'] / df['close'].shift(n)
    
    # Measure Volume Activity
    m = 10  # Lookback period for volume activity
    volume_activity = (df['volume'].rolling(window=m).sum() / m) - df['volume'].shift(1)
    
    # Adjust Intraday Range by Volume Anomaly
    volume_anomaly = df['volume'] - (df['volume'].rolling(window=m).mean())
    adjusted_intraday_range = intraday_range * (1 + volume_anomaly / df['volume'].rolling(window=m).mean())
    
    # Incorporate Price Movement Intensity
    high_low_range = df['high'] - df['low']
    open_close_spread = df['close'] - df['open']
    price_movement_intensity = high_low_range + open_close_spread
    
    # Generate Final Alpha Signal
    final_alpha_signal = (adjusted_intraday_range * volume_anomaly) * momentum * price_movement_intensity
    
    # Identify Volume Spikes
    spike_threshold = df['volume'].rolling(window=20).median() * 2  # Example factor of 2
    spike_indicator = (df['volume'] > spike_threshold).astype(int)
    
    # Adjust for Volume Spike
    adjusted_intraday_range_spike = adjusted_intraday_range * spike_indicator
    
    # Synthesize Intermediate Alpha Factor
    intermediate_alpha_factor = final_alpha_signal * adjusted_intraday_range_spike
    
    # Calculate Daily Price Momentum
    daily_price_momentum = df['close'] - df['close'].shift(1)
    
    # Calculate Short-Term Trend and Volatility
    short_term_trend = daily_price_momentum.ewm(span=5).mean()
    short_term_volatility = daily_price_momentump.rolling(window=5).std()
    short_term_adjusted_trend = short_term_trend / short_term_volatility
    
    # Calculate Long-Term Trend and Volatility
    long_term_trend = daily_price_momentum.ewm(span=20).mean()
    long_term_volatility = daily_price_momentum.rolling(window=20).std()
    long_term_adjusted_trend = long_term_trend / long_term_volatility
    
    # Generate Volume Synchronized Oscillator
    volume_synchronized_oscillator = (short_term_adjusted_trend - long_term_adjusted_trend) * df['volume']
    
    # Integrate Combined Factors
    integrated_factor = (intermediate_alpha_factor * adjusted_intraday_range_spike * daily_price_momentum) * volume_synchronized_oscillator
    
    # Calculate Intraday Volatility
    intraday_volatility = (df['high'] - df['low']) + (df['close'] - df['open'])
    
    # Adjust Intraday Volatility by Volume
    rolling_avg_volume = df['volume'].rolling(window=20).mean()
    adjusted_intraday_volatility = intraday_volatility * (1 + (df['volume'] > rolling_avg_volume).astype(int))
    
    # Incorporate Intraday Volatility into Final Alpha Signal
    final_alpha_signal = integrated_factor * adjusted_intraday_volatility
    
    return final_alpha_signal
