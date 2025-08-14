def heuristics_v2(df):
    # Calculate the Exponential Moving Average (EMA) of Close Prices
    df['EMA_C_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['EMA_C_20'] = df['close'].ewm(span=20, adjust=False).mean()

    # Determine the Rate of Change (ROC) for Close Prices
    df['ROC_C_14'] = df['close'].pct_change(periods=14)

    # Generate the On-Balance Volume (OBV)
    df['OBV'] = (df['close'].diff() / abs(df['close'].diff())) * df['volume']
    df['OBV'] = df['OBV'].fillna(0).cumsum()
    df['OBV_Change'] = df['OBV'].diff()

    # Integrate the Chande Momentum Oscillator (CMO)
    def cmo(data, periods=14):
        data['cmo_up'] = data['close'].diff().apply(lambda x: max(x, 0))
        data['cmo_down'] = data['close'].diff().apply(lambda x: abs(min(x, 0)))
        su = data['cmo_up'].rolling(window=periods).sum()
        sd = data['cmo_down'].rolling(window=periods).sum()
        cmo = ((su - sd) / (su + sd)) * 100
        return cmo
