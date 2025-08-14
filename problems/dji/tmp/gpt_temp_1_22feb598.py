def heuristics_v2(df):
    # Calculate simple price momentum
    df['momentum_5'] = df['close'].pct_change(periods=5)
    df['momentum_10'] = df['close'].pct_change(periods=10)
    df['momentum_20'] = df['close'].pct_change(periods=20)
    df['momentum_60'] = df['close'].pct_change(periods=60)

    # Develop a more complex momentum factor
    df['avg_momentum_5_20'] = (df['momentum_5'] + df['momentum_20']) / 2

    # Measure volume changes over time
    df['volume_change_5'] = df['volume'].pct_change(periods=5)
    df['volume_change_10'] = df['volume'].pct_change(periods=10)
    df['volume_change_20'] = df['volume'].pct_change(periods=20)

    # Analyze the interaction between price and volume
    df['return'] = df['close'].pct_change()
