def heuristics_v2(df):
    # Calculate Intraday Return
    df['Intraday_Range'] = df['high'] - df['low']
    df['Intraday_Return'] = df['Intraday_Range'] / df['close']

    # Calculate Overnight Return
    df['Overnight_Return'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)

    # Incorporate Transaction Amount
    df['Amount_Weighted_Intraday_Return'] = df['Intraday_Return'] * df['amount']
    df['Amount_Weighted_Overnight_Return'] = df['Overnight_Return'] * df['amount']
    total_amount = df['amount'].sum()
    amount_weighted_intraday_return = df['Amount_Weighted_Intraday_Return'].sum() / total_amount
