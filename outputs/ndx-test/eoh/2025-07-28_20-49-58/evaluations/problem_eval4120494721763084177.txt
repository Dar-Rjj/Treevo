def heuristics_v2(df):
    heuristics_matrix = pd.DataFrame(index=df.index)
    heuristics_matrix['Open_Close_Diff'] = df['close'] - df['open']
    heuristics_matrix['High_Low_Diff'] = df['high'] - df['low']
    heuristics_matrix['OC_Ratio'] = df['close'] / df['open']
    heuristics_matrix['HL_Ratio'] = df['high'] / df['low']
    heuristics_matrix['Volume_Change'] = df['volume'].diff()
    return heuristics_matrix
