import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the daily log return
    df['log_return'] = np.log(df['close']).diff()
    
    # Calculate the 10-day Exponential Moving Average (EMA) of log returns to capture short-term momentum
    df['momentum_ema_10d'] = df['log_return'].ewm(span=10, adjust=False).mean()
    
    # Calculate the 30-day standard deviation of log returns to capture volatility
    df['volatility_30d'] = df['log_return'].rolling(window=30).std()
    
    # Calculate the 5-day exponential moving average (EMA) of volume to capture liquidity
    df['volume_ema_5d'] = df['volume'].ewm(span=5, adjust=False).mean()
    
    # Calculate the 14-day Relative Strength Index (RSI) to capture overbought/oversold conditions
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(span=14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
    rs = gain / loss
    df['rsi_14d'] = 100 - (100 / (1 + rs))
    
    # Calculate the 10-day Average True Range (ATR) for volatility
    df['tr_high_low'] = df['high'] - df['low']
    df['tr_high_close_prev'] = (df['high'] - df['close'].shift()).abs()
    df['tr_low_close_prev'] = (df['low'] - df['close'].shift()).abs()
    df['true_range'] = df[['tr_high_low', 'tr_high_close_prev', 'tr_low_close_prev']].max(axis=1)
    df['atr_10d'] = df['true_range'].ewm(span=10, adjust=False).mean()
    
    # Calculate the 20-day and 50-day price change to capture longer-term momentum
    df['price_change_20d'] = (df['close'] / df['close'].shift(20)) - 1
    df['price_change_50d'] = (df['close'] / df['close'].shift(50)) - 1
    
    # Integrate a simple moving average crossover as a technical indicator
    df['sma_50d'] = df['close'].rolling(window=50).mean()
    df['sma_200d'] = df['close'].rolling(window=200).mean()
    df['sma_crossover'] = df['sma_50d'] > df['sma_200d']
    
    # Add seasonality factor
    df['month'] = df.index.month
    seasonal_factor = df['month'].map({1: 1.05, 2: 1.03, 3: 1.07, 4: 1.09, 5: 1.06, 6: 1.08,
                                       7: 1.10, 8: 1.12, 9: 1.11, 10: 1.13, 11: 1.14, 12: 1.15})
    df['seasonal_factor'] = seasonal_factor
    
    # Combine the factors into a single alpha factor with dynamic weighting using a simple linear model
    from sklearn.linear_model import LinearRegression
    X = df[['momentum_ema_10d', 'volatility_30d', 'volume_ema_5d', 'rsi_14d', 'atr_10d',
            'price_change_20d', 'price_change_50d', 'sma_crossover', 'seasonal_factor']].dropna()
    y = df['log_return'].shift(-1).loc[X.index]  # Target is the next day's log return
    model = LinearRegression().fit(X, y)
    df['factor_weights'] = model.predict(df[['momentum_ema_10d', 'volatility_30d', 'volume_ema_5d',
                                             'rsi_14d', 'atr_10d', 'price_change_20d', 'price_change_50d',
                                             'sma_crossover', 'seasonal_factor']])
    
    # Avoid normalization to adhere to the instruction
    return df['factor_weights']
