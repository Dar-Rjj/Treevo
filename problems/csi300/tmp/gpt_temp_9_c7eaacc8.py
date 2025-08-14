def heuristics_v2(df):
    # Hull Moving Average
    def hma(series, period):
        half_period = int(period / 2)
        sqrt_period = int(math.sqrt(period))
        wma_1 = series.rolling(window=half_period).mean()
        wma_2 = series.rolling(window=period).mean()
        diff_wma = 2 * wma_1 - wma_2
        hma = diff_wma.rolling(window=sqrt_period).mean()
        return hma
    
    hma_14 = hma(df['close'], 14)
    hma_50 = hma(df['close'], 50)
    hma_ratio = (hma_14 / hma_50) - 1
    
    # Chaikin Oscillator
    money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    money_flow_volume = money_flow_multiplier * df['volume']
    adl = money_flow_volume.cumsum()
    chaikin_oscillator = adl.rolling(window=3).mean() - adl.rolling(window=10).mean()
    
    # On-Balance Volume (OBV)
    obv = (np.sign(df['close'].diff()) * df['volume']).cumsum()
    
    # Elder Ray Bull Power
    ema_13 = df['close'].ewm(span=13, adjust=False).mean()
    bull_power = df['high'] - ema_13
    
    # Volume weighted typical price
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    volume_weighted = df['volume'] * typical_price
    
    # Composite heuristic
    heuristics_matrix = (hma_ratio + chaikin_oscillator + obv + bull_power + volume_weighted) / 5
    return heuristics_matrix
