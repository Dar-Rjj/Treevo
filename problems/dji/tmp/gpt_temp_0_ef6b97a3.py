def heuristics_v2(df):
    # Momentum-Based Factors
    for period in [5, 10, 20, 50]:
        df[f'SMA_{period}'] = df['close'].rolling(window=period).mean()
        df[f'Close_SMA_{period}_Diff'] = df['close'] - df[f'SMA_{period}']
    
    for period in [7, 30, 90]:  # 1 week, 1 month, 3 months
        df[f'Return_{period}_Days'] = df['close'].pct_change(periods=period)
    
    # Volume-Weighted Price Indicators
    df['VWAP_1D'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['Close_VWAP_1D_Diff'] = df['close'] - df['VWAP_1D']
    
    for period in [5, 10, 20, 50]:
        df[f'VWMA_{period}'] = (df['close'] * df['volume']).rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
        df[f'VWMA_{period}_Slope'] = df[f'VWMA_{period}'].diff()
    
    # Volatility and Range-Based Indicators
    for period in [5, 10, 20, 50]:
        df[f'Volatility_{period}'] = df['close'].pct_change().rolling(window=period).std()
    
    df['True_Range'] = df[['high', 'low']].max(axis=1) - df[['high', 'low']].min(axis=1)
    df['ATR_14'] = df['True_Range'].rolling(window=14).mean()
    
    # Pattern Recognition and Technical Indicator-Based Factors
    def detect_doji(df):
        return ((df['open'] - df['close']).abs() / (df['high'] - df['low']) < 0.1).astype(int)
