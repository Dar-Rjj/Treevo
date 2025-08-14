import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, n=20, m=50, k=10, period=14):
    # Simple daily return
    df['daily_return'] = (df['close'] / df['close'].shift(1)) - 1
    
    # Cumulative logarithmic return over a period
    df['cum_log_return'] = np.log(df['close'] / df['close'].shift(period))
    
    # Daily log return
    df['daily_log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Rolling window volatility
    df['volatility'] = df['daily_log_return'].rolling(window=n).std()
    
    # Volume change
    df['volume_change'] = (df['volume'] / df['volume'].shift(1)) - 1
    
    # Volume-weighted average price (VWAP) with high-low filter
    df['vwap'] = ((df['high'] + df['low']) * df['volume']).rolling(window=n).sum() / df['volume'].rolling(window=n).sum()
    
    # High-low spread with closing price adjustment
    df['high_low_spread'] = (df['high'] - df['low']) / df['close'].shift(1)
    
    # New high and new low indicators with trend analysis
    df['new_high'] = (df['high'] > df['high'].rolling(window=n).max()).astype(int)
    df['new_low'] = (df['low'] < df['low'].rolling(window=n).min()).astype(int)
    
    # Exponential moving average (EMA) crossovers
    df['ema_n'] = df['close'].ewm(span=n, adjust=False).mean()
    df['ema_m'] = df['close'].ewm(span=m, adjust=False).mean()
    df['ema_crossover'] = df['ema_n'] - df['ema_m']
    
    # Rate of Change (ROC) with smoothing
    df['sma_close_k'] = df['close'].rolling(window=k).mean()
    df['roc'] = (df['sma_close_k'] / df['sma_close_k'].shift(n)) - 1
    
    # Enhanced Money Flow Index (MFI)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    positive_money_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0).rolling(window=period).sum()
    negative_money_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0).rolling(window=period).sum()
    df['mfi'] = 100 - (100 / (1 + (positive_money_flow / negative_money_flow)))
    
    # Chaikin Oscillator with volume adjustments
    adl = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']) * df['volume']
    df['chaikin_oscillator'] = adl.ewm(span=3, adjust=False).mean() - adl.ewm(span=10, adjust=False).mean()
    
    # Combine all factors into a single alpha factor
    alpha_factor = (
        df['daily_return'] + 
        df['cum_log_return'] + 
        df['volatility'] + 
        df['volume_change'] + 
        df['vwap'] + 
        df['high_low_spread'] + 
        df['new_high'] + 
        df['new_low'] + 
        df['ema_crossover'] + 
        df['roc'] + 
        df['mfi'] + 
        df['chaikin_oscillator']
    )
    
    return alpha_factor
