import pandas as pd
import pandas as pd

def heuristics_v2(df: pd.DataFrame, sector_data: pd.Series, macroeconomic_data: pd.Series) -> pd.Series:
    # Calculate the daily return
    df['daily_return'] = df['close'].pct_change()
    
    # Short-term momentum using 10-day average of daily returns
    df['momentum_10d'] = df['daily_return'].rolling(window=10).mean()
    
    # Medium-term volatility using 30-day standard deviation of daily returns
    df['volatility_30d'] = df['daily_return'].rolling(window=30).std()
    
    # Short-term liquidity using 5-day exponential moving average (EMA) of volume
    df['volume_ema_5d'] = df['volume'].ewm(span=5, adjust=False).mean()
    
    # Short-term volatility using 10-day average true range (ATR)
    df['tr_high_low'] = df['high'] - df['low']
    df['tr_high_close_prev'] = (df['high'] - df['close'].shift()).abs()
    df['tr_low_close_prev'] = (df['low'] - df['close'].shift()).abs()
    df['true_range'] = df[['tr_high_low', 'tr_high_close_prev', 'tr_low_close_prev']].max(axis=1)
    df['atr_10d'] = df['true_range'].rolling(window=10).mean()
    
    # Longer-term momentum using 20-day price change
    df['price_change_20d'] = (df['close'] / df['close'].shift(20)) - 1
    
    # Very long-term momentum using 50-day price change
    df['price_change_50d'] = (df['close'] / df['close'].shift(50)) - 1
    
    # Incorporate sector-specific indicators
    df['sector_momentum_10d'] = sector_data.rolling(window=10).mean()
    df['sector_volatility_30d'] = sector_data.rolling(window=30).std()
    
    # Incorporate macroeconomic data
    df['macro_economic_indicator'] = macroeconomic_data
    
    # Dynamic weighting based on recent performance
    df['recent_performance'] = df['daily_return'].rolling(window=10).sum()
    df['weight_momentum'] = df['recent_performance'].apply(lambda x: 1 if x > 0 else 0.5)
    df['weight_volatility'] = df['recent_performance'].apply(lambda x: 1.5 if x > 0 else 1)
    
    # Combine the factors into a single alpha factor using dynamic weights and multiplicative interactions
    factor = (
        (df['momentum_10d'] * df['weight_momentum']) * 
        (df['volatility_30d'] * df['weight_volatility']) * 
        df['volume_ema_5d'] * 
        df['atr_10d'] * 
        df['price_change_20d'] * 
        df['price_change_50d'] * 
        df['sector_momentum_10d'] * 
        df['sector_volatility_30d'] * 
        df['macro_economic_indicator']
    )
    return factor
