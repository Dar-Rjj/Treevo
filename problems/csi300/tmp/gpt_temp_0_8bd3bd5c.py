import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate daily returns
    df['daily_return'] = df['close'].pct_change()
    
    # Calculate moving averages
    df['ma_5'] = df['close'].rolling(window=5).mean()
    df['ma_10'] = df['close'].rolling(window=10).mean()
    df['ma_20'] = df['close'].rolling(window=20).mean()
    
    # Trend signal: short-term moving average vs. medium-term moving average
    df['trend_signal'] = df['ma_5'] - df['ma_20']
    
    # Compute True Range and ATR
    df['high_low'] = df['high'] - df['low']
    df['high_close_prev'] = abs(df['high'] - df['close'].shift(1))
    df['low_close_prev'] = abs(df['low'] - df['close'].shift(1))
    df['true_range'] = df[['high_low', 'high_close_prev', 'low_close_prev']].max(axis=1)
    df['atr_14'] = df['true_range'].rolling(window=14).mean()
    
    # Identify price gaps
    df['gap_up'] = (df['open'] > df['close'].shift(1)) * (df['open'] - df['close'].shift(1))
    df['gap_down'] = (df['open'] < df['close'].shift(1)) * (df['open'] - df['close'].shift(1))
    
    # Measure volume changes
    df['volume_change'] = df['volume'].pct_change()
    
    # On-balance volume (OBV)
    obv = [0]
    for i in range(1, len(df)):
        if df['close'][i] > df['close'][i-1]:
            obv.append(obv[-1] + df['volume'][i])
        elif df['close'][i] < df['close'][i-1]:
            obv.append(obv[-1] - df['volume'][i])
        else:
            obv.append(obv[-1])
    df['obv'] = obv
    
    # Price-volume momentum factor
    df['pv_momentum'] = df['daily_return'] * df['volume_change']
    
    # Integrate gap analysis with volume
    df['gap_volume_impact'] = df['gap_up'].fillna(0) * df['volume'] - df['gap_down'].fillna(0) * df['volume']
    
    # Final alpha factor
    df['alpha_factor'] = df['trend_signal'] + df['pv_momentum'] + df['gap_volume_impact']
    
    return df['alpha_factor'].dropna()

# Example usage:
# df = pd.read_csv('your_data.csv', parse_dates=['date'], index_col='date')
# alpha_factors = heuristics_v2(df)
