import pandas as pd
import pandas as pd

def heuristics_v2(df, n=20, m=60, short_term=10, long_term=50):
    # Momentum-based Alpha Factors
    df['PriceMomentum'] = df['close'] / df['close'].shift(n)
    df['VolumeMomentum'] = df['volume'] / df['volume'].shift(n)
    df['AmountMomentum'] = df['amount'] / df['amount'].shift(n)

    # Trend-following Alpha Factors
    df['MA_Crossover_Close'] = df['close'].rolling(window=short_term).mean() - df['close'].rolling(window=long_term).mean()
    df['MA_Crossover_Amount'] = df['amount'].rolling(window=short_term).mean() - df['amount'].rolling(window=long_term).mean()
    df['MA_Crossover_Volume'] = df['volume'].rolling(window=short_term).mean() - df['volume'].rolling(window=long_term).mean()

    # Volatility-based Alpha Factors
    df['Daily_Return'] = df['close'].pct_change()
    df['HistVolatility'] = df['Daily_Return'].rolling(window=n).std()
    df['HighLowVolatility'] = (df['high'] - df['low']).rolling(window=n).mean()
    df['RangeVolatility'] = (df['high'] - df['low']) / df['close']
    df['RangeVolatility'] = df['RangeVolatility'].rolling(window=n).mean()

    # Reversal-based Alpha Factors
    df['ShortTermReversal'] = df['high'].rolling(window=n).max() / df['low'].rolling(window=n).min()
    df['LongTermReversal'] = df['high'].rolling(window=m).max() / df['low'].rolling(window=m).min()

    # Liquidity-based Alpha Factors
    df['DollarVolume'] = df['volume'] * df['close']
    df['AmihudIlliquidity'] = (df['Daily_Return'].abs() / df['DollarVolume']).rolling(window=n).mean()

    # Combine all factors into a single DataFrame
    alpha_factors = df[['PriceMomentum', 'VolumeMomentum', 'AmountMomentum', 
                        'MA_Crossover_Close', 'MA_Crossover_Amount', 'MA_Crossover_Volume',
                        'HistVolatility', 'HighLowVolatility', 'RangeVolatility',
                        'ShortTermReversal', 'LongTermReversal',
                        'DollarVolume', 'AmihudIlliquidity']]

    return alpha_factors
