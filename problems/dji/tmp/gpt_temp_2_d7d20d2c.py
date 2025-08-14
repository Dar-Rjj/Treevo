import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Range
    intraday_range = df['high'] - df['low']
    
    # Calculate Momentum
    n = 5  # Number of days for momentum calculation
    momentum = df['close'].pct_change(periods=n)
    
    # Measure Volume Activity
    m = 10  # Number of days for volume activity calculation
    volume_activity = (df['volume'].rolling(window=m).sum() / m) - df['volume'].shift(1)
    
    # Adjust Intraday Range by Volume Anomaly
    adjusted_intraday_range = intraday_range + (intraday_range * volume_activity)
    
    # Incorporate Price Movement Intensity
    price_movement_intensity = (df['high'] - df['low']) + (df['close'] - df['open'])
    
    # Generate Final Alpha Signal
    final_alpha_signal = (adjusted_intraday_range * volume_activity) * momentum * price_movement_intensity
    
    # Identify Volume Spikes
    spike_threshold = df['volume'].rolling(window=20).median() * 2  # Example factor
    spike_indicator = (df['volume'] > spike_threshold).astype(int)
    
    # Adjust Cumulative Moving Difference
    cumulative_moving_diff = (df['high'] - df['low']).rolling(window=10).sum()
    volume_weighted_avg = (df['volume'].rolling(window=10).mean())
    adjusted_cumulative_moving_diff = cumulative_moving_diff * volume_weighted_avg * spike_indicator
    
    # Synthesize Intermediate Alpha Factor
    intermediate_alpha_factor = final_alpha_signal * adjusted_cumulative_moving_diff
    
    # Calculate Daily Price Momentum
    daily_price_momentum = df['close'] - df['close'].shift(1)
    
    # Calculate Short-Term Trend
    short_term_ema = df['close'].ewm(span=5, adjust=False).mean()
    short_term_volatility = df['close'].rolling(window=5).std()
    short_term_trend = short_term_ema / short_term_volatility
    
    # Calculate Long-Term Trend
    long_term_ema = df['close'].ewm(span=20, adjust=False).mean()
    long_term_volatility = df['close'].rolling(window=20).std()
    long_term_trend = long_term_ema / long_term_volatility
    
    # Generate Volume Synchronized Oscillator
    volume_synchronized_oscillator = (short_term_trend - long_term_trend) * df['volume']
    
    # Integrate Combined Factors
    integrated_factor = (intermediate_alpha_factor * adjusted_intraday_range) + (df['close'] - df['open']) * daily_price_momentum * volume_synchronized_oscillator
    
    # Evaluate Trend Analysis
    trend_analysis = df['close'].rolling(window=5).apply(lambda x: x[-1] > x[0], raw=False).astype(int)
    composite_score = integrated_factor * trend_analysis
    
    return composite_score
