def heuristics_v2(df):
    # Calculate Intraday Range
    df['IntradayRange'] = df['high'] - df['low']

    # Calculate Daily Momentum with Volume Weight
    df['DailyMomentumVolWeight'] = (df['close'] - df['close'].shift(1)) * df['volume']

    # Determine Intraday Reversal Signal
    df['IntradayMove'] = df['high'] - df['close']
    df['TradeIntensity'] = df['volume'] / (df['amount'] / df['volume'])

    # Adjust Momentum by Intraday Volatility and Open-Price Gradient
    df['OpenPriceGradient'] = df['open'] - df['open'].shift(1)
    df['IntradayVolatility'] = df['high'] - df['low']
    df['AdjustedMomentum'] = df['DailyMomentumVolWeight'] / (df['IntradayVolatility'] + df['OpenPriceGradient']).replace(0, 1)

    # Identify Volume Spikes
    m = 20
    df['AvgVolume'] = df['volume'].rolling(window=m).mean()
    df['VolumeSpike'] = df['volume'] > df['AvgVolume'] * 1.5

    # Adjust Momentum by Volume Spike
    df['AdjustedMomentum'] = df['AdjustedMomentum'] * (1 + 0.5 * df['VolumeSpike'])

    # Weight Intraday Move by Trade Intensity
    df['WeightedIntradayMove'] = df['IntradayMove'] * df['TradeIntensity']

    # Incorporate Intraday Range Expansion
    df['PrevIntradayRange'] = df['IntradayRange'].shift(1)
    df['RangeExpansion'] = df['IntradayRange'] > df['PrevIntradayRange']
    df['AdjustedIntradayMove'] = df['WeightedIntradayMove'] * (1 + 0.5 * df['RangeExpansion'])

    # Price-based Momentum
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA21'] = df['close'].rolling(window=21).mean()
    df['PriceMomentum'] = df['MA5'] - df['MA21']

    # Volume-based Momentum
    df['VolMA5'] = df['volume'].rolling(window=5).mean()
    df['VolMA21'] = df['volume'].rolling(window=21).mean()
    df['VolumeMomentum'] = df['VolMA5'] - df['VolMA21']

    # Price Oscillators
    df['HighSMA10'] = df['high'].rolling(window=10).mean()
    df['LowSMA10'] = df['low'].rolling(window=10).mean()
    df['PriceOscillator'] = df['HighSMA10'] - df['LowSMA10']

    # Volatility Indicators
    df['TrueRange'] = df[['high' - 'low', 'high' - 'close'.shift(1), 'low' - 'close'.shift(1)]].abs().max(axis=1)
    df['ATR14'] = df['TrueRange'].rolling(window=14).mean()

    # Reversal Indicators (3-day RSI)
    def rsi(series, n):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=n).mean()
        avg_loss = loss.rolling(window=n).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
