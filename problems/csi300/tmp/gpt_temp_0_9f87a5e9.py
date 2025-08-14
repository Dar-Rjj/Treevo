import pandas as pd
import pandas as pd

def heuristics_v2(df: pd.DataFrame, sector_trends: pd.Series, macroeconomic_indicators: pd.DataFrame) -> pd.Series:
    # Calculate the daily return
    df['daily_return'] = df['close'].pct_change()
    
    # Adaptive window sizes based on volatility
    adaptive_window_10d = (df['close'].rolling(window=30).std() * 10).astype(int).clip(lower=5, upper=20)
    adaptive_window_30d = (df['close'].rolling(window=60).std() * 30).astype(int).clip(lower=15, upper=60)
    
    # Calculate the adaptive 10-day average of daily returns to capture short-term momentum
    df['momentum_adaptive_10d'] = df['daily_return'].rolling(window=adaptive_window_10d).mean()
    
    # Calculate the adaptive 30-day standard deviation of daily returns to capture volatility
    df['volatility_adaptive_30d'] = df['daily_return'].rolling(window=adaptive_window_30d).std()
    
    # Calculate the 5-day exponential moving average (EMA) of volume to capture liquidity
    df['volume_ema_5d'] = df['volume'].ewm(span=5, adjust=False).mean()
    
    # Calculate the adaptive 10-day average true range (ATR) for volatility
    df['tr_high_low'] = df['high'] - df['low']
    df['tr_high_close_prev'] = (df['high'] - df['close'].shift()).abs()
    df['tr_low_close_prev'] = (df['low'] - df['close'].shift()).abs()
    df['true_range'] = df[['tr_high_low', 'tr_high_close_prev', 'tr_low_close_prev']].max(axis=1)
    df['atr_adaptive_10d'] = df['true_range'].rolling(window=adaptive_window_10d).mean()
    
    # Calculate the adaptive 20-day price change to capture longer-term momentum
    df['price_change_adaptive_20d'] = (df['close'] / df['close'].shift(adaptive_window_10d * 2)) - 1
    
    # Calculate the adaptive 50-day price change to capture even longer-term trends
    df['price_change_adaptive_50d'] = (df['close'] / df['close'].shift(adaptive_window_30d * 2)).fillna(0) - 1
    
    # Integrate a simple moving average crossover as a technical indicator with adaptive windows
    df['sma_adaptive_50d'] = df['close'].rolling(window=adaptive_window_30d).mean()
    df['sma_adaptive_200d'] = df['close'].rolling(window=adaptive_window_30d * 4).mean()
    df['sma_crossover_adaptive'] = (df['sma_adaptive_50d'] > df['sma_adaptive_200d']).astype(int)
    
    # Incorporate sector-specific trends
    df['sector_trend'] = sector_trends.reindex(df.index, method='ffill').fillna(method='bfill')
    
    # Incorporate macroeconomic indicators
    df = df.merge(macroeconomic_indicators, left_index=True, right_index=True, how='left')
    
    # Combine the factors into a single alpha factor
    factor = (
        df['momentum_adaptive_10d'] * 
        df['volatility_adaptive_30d'] * 
        df['volume_ema_5d'] * 
        df['atr_adaptive_10d'] * 
        df['price_change_adaptive_20d'] * 
        df['price_change_adaptive_50d'] * 
        df['sma_crossover_adaptive'] * 
        df['sector_trend'] * 
        df['macro_indicator_1'] * 
        df['macro_indicator_2']
    )
    
    # Return the factor
    return factor
