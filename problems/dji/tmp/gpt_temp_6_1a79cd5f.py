def heuristics_v2(df):
    def roc(series, periods=20):
        return (series - series.shift(periods)) / series.shift(periods)
    
    def dmi(high, low, close, period=14):
        high_diff = high.diff()
        low_diff = -low.diff()
        
        positive_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        negative_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        plus_di = 100 * (positive_dm.rolling(window=period).sum() / atr)
        minus_di = 100 * (negative_dm.rolling(window=period).sum() / atr)
        
        dmi = (plus_di - minus_di).rename('dmi')
        return dmi

    roc_close = roc(df['close'])
    dmi_value = dmi(df['high'], df['low'], df['close'])
    combined_factor = (roc_close + dmi_value).rename('combined_factor')
    heuristics_matrix = combined_factor.rolling(window=20).mean().rename('heuristic_factor')

    return heuristics_matrix
