import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Move
    df['IntradayMove'] = df['high'] - df['close']
    
    # Calculate Intraday Volatility
    df['IntradayVolatility'] = df['high'] - df['low']
    
    # Calculate Adjusted Daily Momentum
    df['DailyMomentum'] = df['close'] - df['close'].shift(1)
    df['AdjustedDailyMomentum'] = df['DailyMomentum'] / df['IntradayVolatility']
    
    # Estimate Trade Intensity
    df['TradeIntensity'] = df['volume'] / ((df['high'] + df['low']) / 2)
    
    # Weight Intraday Move by Trade Intensity
    df['WeightedIntradayMove'] = df['IntradayMove'] * df['TradeIntensity']
    
    # Weight Adjusted Daily Momentum by Trade Intensity
    df['WeightedAdjustedDailyMomentum'] = df['AdjustedDailyMomentum'] * df['TradeIntensity']
    
    # Calculate Intraday Reversal
    df['IntradayReversal'] = df['high'] - df['open']
    
    # Weight Intraday Reversal by Trade Intensity
    df['WeightedIntradayReversal'] = df['IntradayReversal'] * df['TradeIntensity']
    
    # Calculate Intraday Gap
    df['IntradayGap'] = df['open'] - df['close'].shift(1)
    
    # Weight Intraday Gap by Trade Intensity
    df['WeightedIntradayGap'] = df['IntradayGap'] * df['TradeIntensity']
    
    # Calculate Intraday Midpoint
    df['IntradayMidpoint'] = (df['high'] + df['low']) / 2
    
    # Calculate Intraday Range Ratio
    df['IntradayRangeRatio'] = (df['high'] - df['low']) / df['IntradayMidpoint']
    
    # Weight Intraday Range Ratio by Trade Intensity
    df['WeightedIntradayRangeRatio'] = df['IntradayRangeRatio'] * df['TradeIntensity']
    
    # Calculate the 5-day and 21-day moving average of the close price
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA21'] = df['close'].rolling(window=21).mean()
    
    # Compute the difference between the 5-day and 21-day moving averages
    df['TrendMomentum'] = df['MA5'] - df['MA21']
    
    # Calculate the 10-day simple moving average (SMA) of the high and low prices
    df['SMA10High'] = df['high'].rolling(window=10).mean()
    df['SMA10Low'] = df['low'].rolling(window=10).mean()
    
    # Compute the difference between the 10-day SMA of high and low prices
    df['PriceOscillator'] = df['SMA10High'] - df['SMA10Low']
    
    # Calculate the Average True Range (ATR) over a 14-day period
    df['TrueRange'] = df[['high' - df['low'], df['high'] - df['close'].shift(1), df['close'].shift(1) - df['low']]].max(axis=1)
    df['ATR'] = df['TrueRange'].rolling(window=14).mean()
    
    # Combine All Weighted Components with Trend and Volatility Indicators
    df['AlphaFactor'] = (
        df['WeightedIntradayMove'] +
        df['WeightedAdjustedDailyMomentum'] +
        df['WeightedIntradayReversal'] +
        df['WeightedIntradayGap'] +
        df['WeightedIntradayRangeRatio'] +
        df['TrendMomentum'] +
        df['PriceOscillator'] +
        df['ATR']
    )
    
    return df['AlphaFactor'].dropna()
