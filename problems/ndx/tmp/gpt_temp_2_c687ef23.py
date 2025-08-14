import pandas as pd

def heuristics_v2(df):
    # Calculate the Average True Range (ATR) for a 14-day period
    df['H-L'] = abs(df['high'] - df['low'])
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    atr = df['TR'].rolling(window=14).mean()
    
    # Calculate the Money Flow Index (MFI) for a 14-day period
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    positive_money_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_money_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
    mfi_ratio = positive_money_flow.rolling(window=14).sum() / negative_money_flow.rolling(window=14).sum()
    mfi = 100 - (100 / (1 + mfi_ratio))

    # Calculate daily returns
    df['return'] = df['close'].pct_change()
    
    # Drop rows with NaN values in atr, mfi, and return
    valid_df = df.dropna(subset=['TR', 'mfi', 'return'])
    
    # Compute the correlation of ATR and MFI with the daily returns
    atr_corr = valid_df['TR'].corr(valid_df['return'])
    mfi_corr = valid_df['mfi'].corr(valid_df['return'])
    
    # Calculate the weighting factors
    total_corr = abs(atr_corr) + abs(mfi_corr)
    atr_weight = abs(atr_corr) / total_corr
    mfi_weight = abs(mfi_corr) / total_corr
    
    # Combine ATR and MFI into a single heuristic measure using the computed weights
    heuristics_matrix = (atr * atr_weight) + (mfi * mfi_weight)
    
    return heuristics_matrix
