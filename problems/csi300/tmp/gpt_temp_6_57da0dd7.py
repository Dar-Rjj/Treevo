import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate simple moving averages for 5 and 20 days
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()

    # Calculate the Relative Strength (RS) as the ratio of the current close to the 20-day SMA
    df['RS'] = df['close'] / df['SMA_20']

    # Calculate the price change and volume change
    df['price_change'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()

    # Calculate the volatility as the standard deviation of the last 5 days' closing prices
    df['volatility'] = df['close'].rolling(window=5).std()

    # Normalize the RS by the volatility
    df['normalized_RS'] = df['RS'] / (df['volatility'] + 1e-7)

    # Integrate market context and liquidity
    df['market_context'] = (df['high'] - df['low']) / df['close']
    df['liquidity'] = df['amount'] / df['volume']

    # Adaptive weights for different factors
    df['weight_RS'] = df['normalized_RS'].rolling(window=20).corr(df['price_change'])
    df['weight_price_change'] = df['price_change'].rolling(window=20).corr(df['price_change'])
    df['weight_volume_change'] = df['volume_change'].rolling(window=20).corr(df['price_change'])

    # Dynamic factors
    df['dynamic_momentum'] = (df['SMA_5'] - df['SMA_20']) / (df['SMA_20'] + 1e-7)
    df['dynamic_volatility'] = (df['close'] - df['SMA_20']) / (df['volatility'] + 1e-7)

    # Additional dynamic factors
    df['dynamic_market_context'] = (df['high'].rolling(window=5).mean() - df['low'].rolling(window=5).mean()) / (df['close'].rolling(window=20).mean() + 1e-7)
    df['dynamic_liquidity'] = df['amount'].rolling(window=5).sum() / (df['volume'].rolling(window=5).sum() + 1e-7)

    # Adaptive weights for additional dynamic factors
    df['weight_dynamic_market_context'] = df['dynamic_market_context'].rolling(window=20).corr(df['price_change'])
    df['weight_dynamic_liquidity'] = df['dynamic_liquidity'].rolling(window=20).corr(df['price_change'])

    # Combine the normalized RS, price change, and volume change with adaptive weights to form the alpha factor
    df['alpha_factor'] = (
        df['weight_RS'] * df['normalized_RS'] + 
        df['weight_price_change'] * df['price_change'] + 
        df['weight_volume_change'] * 2 * df['volume_change'] + 
        df['weight_dynamic_momentum'] * df['dynamic_momentum'] + 
        df['weight_dynamic_volatility'] * df['dynamic_volatility'] +
        df['weight_dynamic_market_context'] * df['dynamic_market_context'] + 
        df['weight_dynamic_liquidity'] * df['dynamic_liquidity'] +
        df['market_context'] + 
        df['liquidity']
    ) / 10

    return df['alpha_factor']
