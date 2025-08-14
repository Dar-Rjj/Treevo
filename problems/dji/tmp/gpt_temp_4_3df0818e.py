import pandas as pd
import pandas as pd

def heuristics_v2(df, n=14):
    # Momentum of closing prices
    df['momentum'] = df['close'].pct_change().rolling(window=n).mean()
    
    # Average True Range (ATR)
    df['tr'] = df[['high', 'low', 'close']].shift(1).apply(lambda x: max(x['high'] - x['low'], abs(x['high'] - df['close']), abs(x['low'] - df['close'])), axis=1)
    df['atr'] = df['tr'].rolling(window=n).mean()
    
    # Bull/Bear strength indicator
    df['bull_bear'] = (df['close'] - df['open']).rolling(window=n).mean()
    
    # Volume shock factor
    df['volume_shock'] = df['volume'] - df['volume'].rolling(window=n).median()
    
    # On-Balance Volume (OBV)
    df['obv'] = (df['close'] > df['close'].shift(1)).astype(int) * df['volume'] - (df['close'] < df['close'].shift(1)).astype(int) * df['volume']
    df['obv'] = df['obv'].cumsum()
    
    # Money Flow Index (MFI) alternative
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['money_flow'] = df['typical_price'] * df['volume']
    df['positive_money_flow'] = df[df['typical_price'] > df['typical_price'].shift(1)]['money_flow']
    df['negative_money_flow'] = df[df['typical_price'] < df['typical_price'].shift(1)]['money_flow']
    df['mfi'] = df['positive_money_flow'].rolling(window=n).sum() / (df['positive_money_flow'].rolling(window=n).sum() + df['negative_money_flow'].rolling(window=n).sum())
    
    # Fractal efficiency
    df['net_movement'] = df['close'] - df['open'].shift(n-1)
    df['total_movement'] = (df['high'] - df['low']).rolling(window=n).sum()
    df['fractal_efficiency'] = df['net_movement'] / df['total_movement']
    
    # Gap analysis
    df['gap'] = df['open'] - df['close'].shift(1)
    df['gap_analysis'] = df['gap'].rolling(window=n).mean()
    
    # Support/Resistance breakout
    df['support'] = df['low'].rolling(window=n).min()
    df['resistance'] = df['high'].rolling(window=n).max()
    df['breakout'] = (df['close'] > df['resistance']).astype(int) - (df['close'] < df['support']).astype(int)
    
    # Combine all factors into a single alpha factor
    df['alpha_factor'] = (
        df['momentum'] + 
        df['atr'] + 
        df['bull_bear'] + 
        df['volume_shock'] + 
        df['obv'] + 
        df['mfi'] + 
        df['fractal_efficiency'] + 
        df['gap_analysis'] + 
        df['breakout']
    )
    
    return df['alpha_factor'].dropna()
