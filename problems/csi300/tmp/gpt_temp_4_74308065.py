def heuristics_v2(df):
    # Intraday Momentum and Volume-Weighted Price Trends
    df['High_Low_Ratio'] = df['high'] / df['low']
    df['Open_Close_Diff'] = df['close'] - df['open']
    df['Volume_Weighted_Momentum'] = df['Open_Close_Diff'] * (df['volume'] / df['volume'].rolling(window=20).mean())
    df['VWAP'] = (df['amount'] / df['volume']).rolling(window=5).mean()
    df['VWAP_Closing_Price_Diff'] = df['VWAP'] - df['close']

    # Day-to-Day and Long-Term Momentum
    df['Prev_Close'] = df['close'].shift(1)
    df['Yesterdays_Close_to_Todays_Open'] = df['open'] - df['Prev_Close']
    df['Short_Term_Momentum'] = df['close'] - df['close'].rolling(window=7).mean()
    df['Long_Term_Momentum'] = df['close'] - df['close'].rolling(window=25).mean()
    df['Momentum_Differential'] = df['Long_Term_Momentum'] - df['Short_Term_Momentum']

    # True Range and Average True Range
    df['True_Range'] = df[['high', 'low', 'close']].apply(
        lambda x: max(x['high'] - x['low'], abs(x['high'] - df['close'].shift(1)), abs(x['low'] - df['close'].shift(1))), axis=1)
    df['ATR'] = df['True_Range'].rolling(window=14).mean()
    df['Price_Volatility_Factor'] = df['ATR'] / df['close'].rolling(window=1).mean()

    # Volume Patterns and Gaps
    df['Volume_Change'] = df['volume'] - df['volume'].shift(1)
    df['Cumulative_Volume'] = df['volume'].rolling(window=10).sum()
    df['Volume_Spike'] = df['volume'] > df['volume'].rolling(window=20).mean() * 1.5
    df['Gap_Size'] = df['open'] - df['Prev_Close']
    df['Large_Gap_Flag'] = (df['Gap_Size'] > df['ATR'] * 2).astype(int)

    # Open to Close Price Movement
    df['Open_Close_Pct_Change'] = (df['close'] - df['open']) / df['open']
    df['Consistency_Factor'] = df['Open_Close_Pct_Change'].rolling(window=5).mean()

    # Integrate Signals
    df['Volume_Adjusted_Momentum'] = df['Volume_Weighted_Momentum'] * df['Momentum_Differential']
    df['Significant_Volume_Increase'] = df['volume'] > df['volume'].rolling(window=20).mean() * 1.5
    df.loc[df['Significant_Volume_Increase'], 'Volume_Adjusted_Momentum'] *= 1.5

    # Enhance Momentum and Volume Interactions
    df['Volume_Ratio'] = df['volume'].rolling(window=5).mean() / df['volume'].rolling(window=20).mean()
    df['Adjusted_Momentum_Differential'] = df['Momentum_Differential'] * df['Volume_Ratio']
    df['Combined_Momentum'] = df['Volume_Adjusted_Momentum'] + df['Adjusted_Momentum_Differential']

    # Incorporate Gap Analysis into Momentum
    df['Gap_Adjusted_Momentum_Differential'] = df['Momentum_Differential'] + (df['Gap_Size'] / df['ATR'])
    df['Combined_Momentum_with_Gap'] = df['Volume_Adjusted_Momentum'] + df['Gap_Adjusted_Momentum_Differential']

    # Explore Volume-Weighted Price Oscillations
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['Adjusted_RSI'] = df['RSI'] * (df['VWAP'] / df['close'])
    df['Combined_Momentum_with_RSI'] = df['Volume_Adjusted_Momentum'] + df['Adjusted_RSI']

    # Final Factor
    factor = df['Combined_Momentum_with_RSI']
    return factor
