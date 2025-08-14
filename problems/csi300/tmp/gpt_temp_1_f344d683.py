def heuristics_v2(df):
    # Calculate VWAP
    df['TypicalPrice'] = (df['high'] + df['low'] + df['close']) / 3
    total_volume = df['volume'].sum()
    vwap = (df['TypicalPrice'] * df['volume']).sum() / total_volume
    
    # Calculate Momentum
    df['10DayPriceChange'] = df['close'] - df['close'].shift(10)
    df['60DayPriceChange'] = df['close'] - df['close'].shift(60)
    df['ShortTermMomentum'] = df['close'].pct_change().rolling(window=5).mean()
    df['MediumTermMomentum'] = df['close'].pct_change().rolling(window=20).mean()
    
    # Calculate Daily Returns
    df['DailyReturn'] = df['close'].pct_change()
    df['IntradayReturn'] = (df['close'] - df['open']) / df['open']
    df['HighLowRangeReturn'] = (df['high'] - df['low']) / df['low']
    
    # Measure Liquidity
    avg_volume_20 = df['volume'].rolling(window=20).mean()
    df['LiquidityIndicator'] = df['volume'] / avg_volume_20
    
    # Examine Volume and Amount in Relation to Returns
    df['VolumeChange'] = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    df['AmountChange'] = (df['amount'] - df['amount'].shift(1)) / df['amount'].shift(1)
    df['ReturnAdjustedByVolumeChange'] = (df['close'] - df['close'].shift(1)) / (df['volume'] - df['volume'].shift(1))
    
    # Calculate Volatility
    df['10DayStdDev'] = df['close'].pct_change().rolling(window=10).std()
    df['60DayStdDev'] = df['close'].pct_change().rolling(window=60).std()
    df['VolatilityRelativeToIntradayRange'] = df['close'].pct_change().rolling(window=30).std() / (df['high'] - df['low'])
    
    # Combine Momentum, Volatility, and Liquidity
    df['AdjustedShortTermProduct'] = df['10DayPriceChange'] * df['10DayStdDev'] * df['LiquidityIndicator']
    df['AdjustedLongTermProduct'] = df['60DayPriceChange'] * df['60DayStdDev'] * df['LiquidityIndicator']
    
    # VWAP-Momentum-Liquidity Composite
    df['VWAPAdjustedShortTermMomentum'] = (vwap - df['open']) * df['AdjustedShortTermProduct']
    df['VWAPAdjustedLongTermMomentum'] = (vwap - df['close']) * df['AdjustedLongTermProduct']
    
    # Introduce Open-Close Price Spread
    df['OpenCloseSpread'] = df['close'] - df['open']
    df['AdjustedOpenCloseSpread'] = df['OpenCloseSpread'] * df['LiquidityIndicator']
    
    # Additional Factor: VWAP-Open-Close Spread
    df['VWAPOpenSpread'] = vwap - df['open']
    df['VWAPCloseSpread'] = vwap - df['close']
    df['VWAPOpenSpreadAdjusted'] = df['VWAPOpenSpread'] * df['LiquidityIndicator']
    df['VWAPCloseSpreadAdjusted'] = df['VWAPCloseSpread'] * df['LiquidityIndicator']
    
    # Evaluate Market Sentiment through Price Behavior
    bearish_sentiment = (df['close'] < df['open']).rolling(window=5).sum()
    bullish_sentiment = (df['close'] > df['open']).rolling(window=5).sum()
    df['SentimentStrength'] = (bullish_sentiment - bearish_sentiment) / 5
    
    # Explore Technical Indicators
    short_term_ma = df['close'].rolling(window=9).mean()
    long_term_ma = df['close'].rolling(window=26).mean()
    df['MovingAverageCrossover'] = short_term_ma - long_term_ma
    df['MomentumIndicator'] = (df['close'] / df['close'].shift(14)) - 1
    daily_returns = df['close'].pct_change()
