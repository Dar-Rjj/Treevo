import pandas as pd
def heuristics_v2(df: pd.DataFrame, sector: pd.Series) -> pd.Series:
    # Calculate the daily return
    df['daily_return'] = df['close'].pct_change()
    
    # Calculate the 10-day average of daily returns to capture short-term momentum
    df['momentum_10d'] = df['daily_return'].rolling(window=10).mean()
    
    # Calculate the 30-day standard deviation of daily returns to capture volatility
    df['volatility_30d'] = df['daily_return'].rolling(window=30).std()
    
    # Calculate the 5-day exponential moving average (EMA) of volume to capture liquidity
    df['volume_ema_5d'] = df['volume'].ewm(span=5, adjust=False).mean()
    
    # Calculate the 10-day average true range (ATR) for volatility
    df['tr_high_low'] = df['high'] - df['low']
    df['tr_high_close_prev'] = (df['high'] - df['close'].shift()).abs()
    df['tr_low_close_prev'] = (df['low'] - df['close'].shift()).abs()
    df['true_range'] = df[['tr_high_low', 'tr_high_close_prev', 'tr_low_close_prev']].max(axis=1)
    df['atr_10d'] = df['true_range'].rolling(window=10).mean()
    
    # Calculate the 20-day price change to capture longer-term momentum
    df['price_change_20d'] = (df['close'] / df['close'].shift(20)) - 1
    
    # Calculate the 50-day price change to capture even longer-term trends
    df['price_change_50d'] = (df['close'] / df['close'].shift(50)) - 1
    
    # Define sector-specific weights
    sector_weights = {
        'Technology': [0.2, 0.15, 0.15, 0.15, 0.15, 0.2],
        'Healthcare': [0.2, 0.2, 0.1, 0.1, 0.2, 0.2],
        'Financials': [0.1, 0.1, 0.2, 0.2, 0.2, 0.2],
        'Consumer': [0.15, 0.15, 0.15, 0.15, 0.15, 0.25],
        'Energy': [0.1, 0.1, 0.2, 0.2, 0.2, 0.2],
        'Other': [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667]
    }
    
    # Apply sector-specific weights
    factors = [df['momentum_10d'], df['volatility_30d'], df['volume_ema_5d'], df['atr_10d'], df['price_change_20d'], df['price_change_50d']]
    weighted_factors = [f * sector_weights.get(sector.name, sector_weights['Other'])[i] for i, f in enumerate(factors)]
    
    # Combine the weighted factors into a single alpha factor
    factor = sum(weighted_factors)
    return factor
