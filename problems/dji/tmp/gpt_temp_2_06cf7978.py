import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate simple moving averages
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()

    # Calculate price changes over different time horizons
    df['daily_price_change'] = df['close'].diff()
    df['weekly_price_change'] = df['close'] - df['close'].shift(5)
    df['monthly_price_change'] = df['close'] - df['close'].shift(20)

    # Calculate historical volatility
    daily_returns = df['close'].pct_change()
    df['volatility_5'] = daily_returns.rolling(window=5).std() * (252 ** 0.5)  # Annualize
    df['volatility_20'] = daily_returns.rolling(window=20).std() * (252 ** 0.5)  # Annualize

    # Combine momentum and volatility
    df['momentum_to_volatility_ratio'] = df['monthly_price_change'] / df['volatility_20']
    df['weekly_momentum_adjusted_for_5d_volatility'] = df['weekly_price_change'] / df['volatility_5']

    # Examine trading volume and amount
    df['avg_volume_5'] = df['volume'].rolling(window=5).mean()
    df['avg_volume_20'] = df['volume'].rolling(window=20).mean()

    df['daily_volume_change'] = df['volume'].diff()
    df['weekly_volume_change'] = df['volume'] - df['volume'].shift(5)
    df['monthly_volume_change'] = df['volume'] - df['volume'].shift(20)

    df['daily_trading_amount'] = df['amount']
    df['weekly_trading_amount'] = df['amount'].rolling(window=5).sum()
    df['monthly_trading_amount'] = df['amount'].rolling(window=20).sum()

    # Analyze high, low, and range
    df['daily_price_range'] = df['high'] - df['low']
    df['true_range'] = df.apply(
        lambda row: max(row['high'] - row['low'], abs(row['high'] - row['close'].shift(1)), abs(row['low'] - row['close'].shift(1))),
        axis=1
    )
    df['ATR_5'] = df['true_range'].rolling(window=5).mean()
    df['ATR_20'] = df['true_range'].rolling(window=20).mean()

    df['std_high_5'] = df['high'].rolling(window=5).std()
    df['std_low_5'] = df['low'].rolling(window=5).std()

    # Examine opening and closing price relationships
    df['open_to_close_difference'] = df['close'] - df['open']
    df['gap'] = df['open'] - df['close'].shift(1)
    df['open_to_high_ratio'] = df['open'] / df['high']
    df['close_to_low_ratio'] = df['close'] / df['low']

    # Generate composite factors
    df['momentum_to_volume_ratio'] = df['monthly_price_change'] / df['avg_volume_20']
    df['volume_adjusted_weekly_momentum'] = df['weekly_price_change'] * (df['avg_volume_5'] / df['avg_volume_20'])

    df['range_to_volatility_ratio'] = df['daily_price_range'] / df['volatility_5']
    df['true_range_to_ATR_ratio'] = df['true_range'] / df['ATR_20']

    df['open_to_close_diff_adjusted_by_volume'] = df['open_to_close_difference'] * (df['volume'] / df['avg_volume_20'])
    df['gap_adjusted_by_volume'] = df['gap'] * (df['volume'] / df['avg_volume_20'])

    # Return the final alpha factor
    return df['momentum_to_volatility_ratio']  # Example output, you can choose any other factor
