import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import yfinance as yf

def heuristics_v2(df: pd.DataFrame, external_economic_indicators: pd.DataFrame) -> pd.Series:
    # Calculate the daily return
    df['daily_return'] = df['close'].pct_change()
    
    # Calculate the 10-day and 30-day Exponential Moving Average (EMA) of close prices
    df['ema_10d'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema_30d'] = df['close'].ewm(span=30, adjust=False).mean()
    
    # Calculate the 14-day Relative Strength Index (RSI)
    delta = df['close'].diff(1)
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['rsi_14d'] = 100 - (100 / (1 + rs))
    
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
    
    # Integrate a simple moving average crossover as a technical indicator
    df['sma_50d'] = df['close'].rolling(window=50).mean()
    df['sma_200d'] = df['close'].rolling(window=200).mean()
    df['sma_crossover'] = df['sma_50d'] > df['sma_200d']
    
    # Seasonality adjustment using seasonal_decompose
    decomposition = seasonal_decompose(df['close'], model='additive', period=252)
    df['seasonal_adjusted_close'] = df['close'] - decomposition.seasonal
    
    # Incorporate external economic indicators
    df = df.join(external_economic_indicators, how='left')
    
    # Combine the factors into a single alpha factor
    factor = (
        df['ema_10d'] * df['ema_30d'] * df['rsi_14d'] * 
        df['volume_ema_5d'] * df['atr_10d'] * 
        df['price_change_20d'] * df['price_change_50d'] * 
        df['sma_crossover'].astype(int) * df['seasonal_adjusted_close'] * 
        df[external_economic_indicators.columns].fillna(method='ffill').sum(axis=1)
    )
    
    # Handle NaN values
    factor = factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return factor
