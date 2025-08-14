import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Price Movement
    df['high_low_range'] = df['high'] - df['low']
    df['close_open_diff'] = df['close'] - df['open']
    
    # Incorporate Volume Influence
    df['volume_adjusted_momentum'] = df['total_volume'] * (df['close_open_diff'] / df['high_low_range'])
    df['sma_10_vol_adj_momentum'] = df['volume_adjusted_momentum'].rolling(window=10).mean()
    
    # Adjust for Market Volatility
    df['daily_return'] = df['close'].pct_change()
    df['market_volatility'] = df['daily_return'].rolling(window=30).std()
    df['adjusted_momentum'] = df['sma_10_vol_adj_momentum'] - df['market_volatility']
    
    # Incorporate Trend Reversal Signal
    df['sma_5_close'] = df['close'].rolling(window=5).mean()
    df['sma_20_close'] = df['close'].rolling(window=20).mean()
    df['momentum_reversal'] = df['sma_5_close'] - df['sma_20_close']
    
    # Integrate Reversal Signal into Final Factor
    df['reversal_signal'] = np.where(df['momentum_reversal'] > 0, 1, -1)
    df['final_alpha_factor'] = df['adjusted_momentum'] + df['reversal_signal']
    
    return df['final_alpha_factor']
