import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the difference between high and low prices as a measure of daily price range
    daily_range = df['high'] - df['low']
    
    # Calculate the difference between close and open prices as a measure of daily price movement
    daily_movement = df['close'] - df['open']
    
    # Create a factor that scales the daily movement by the daily range
    # This factor indicates the relative size of the daily movement in the context of the daily range
    movement_factor = daily_movement / (daily_range + 1e-7)
    
    # Calculate the 5-day return to capture momentum
    five_day_return = df['close'].pct_change(periods=5)
    
    # Calculate the 5-day average volume to capture volume momentum
    five_day_avg_volume = df['volume'].rolling(window=5).mean()
    
    # Create a factor that scales the current volume by the 5-day average volume
    # This factor indicates the relative strength of the current volume compared to the recent average
    volume_factor = df['volume'] / (five_day_avg_volume + 1e-7)
    
    # Calculate the 30-day volatility to capture market volatility
    thirty_day_volatility = df['close'].rolling(window=30).std()
    
    # Create an adaptive window for the moving average based on the 30-day volatility
    # Lower volatility leads to a longer window, higher volatility leads to a shorter window
    adaptive_window = 30 - (thirty_day_volatility * 10).astype(int)
    adaptive_window = adaptive_window.clip(lower=10, upper=30)
    
    # Calculate the adaptive moving average
    adaptive_ma = df['close'].rolling(window=adaptive_window).mean()
    
    # Create a factor that measures the distance of the current close price from the adaptive moving average
    adaptive_factor = (df['close'] - adaptive_ma) / (adaptive_ma + 1e-7)
    
    # Combine the movement, momentum, volume, and adaptive factors
    # The weights (0.3, 0.2, 0.2, 0.3) are chosen to balance the importance of each factor
    factor = 0.3 * movement_factor + 0.2 * five_day_return + 0.2 * volume_factor + 0.3 * adaptive_factor
    
    return factor
