import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    import pandas as pd
    import numpy as np

    # Calculate the positive and negative part of the amount over volume
    df['positive_amount_vol'] = (df['amount'] / df['volume']).clip(lower=0)
    df['negative_amount_vol'] = (df['amount'] / df['volume']).clip(upper=0)

    # Sum of positive and absolute negative parts over a rolling window
    pos_sum = df['positive_amount_vol'].rolling(window=5).sum()
    neg_sum_abs = df['negative_amount_vol'].abs().rolling(window=5).sum()

    # Factor: ratio of positive sum to absolute negative sum
    factor1 = pos_sum / (neg_sum_abs + 1e-7)

    # Calculate rolling standard deviation of the closing price for volatility
    volatility = df['close'].rolling(window=5).std()

    # Calculate the momentum of the closing price
    momentum = df['close'].pct_change(periods=5).rolling(window=5).mean()

    # Calculate the liquidity as the average daily volume
    liquidity = df['volume'].rolling(window=5).mean()

    # Calculate the normalized return using the range (high - low) as the denominator
    daily_range = df['high'] - df['low']
    norm_return = df['close'].pct_change() / daily_range

    # Combine the factors: momentum, volatility, and liquidity
    combined_factor = (momentum / (volatility + 1e-7)) * (liquidity / (daily_range + 1e-7))

    # Incorporate seasonality by adding a sine and cosine transformation of the date
    df['date'] = pd.to_datetime(df.index)
    df['month_sin'] = np.sin(2 * np.pi * df['date'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['date'].dt.month / 12)

    # Advanced statistical method: add a moving average cross-over
    short_window = 5
    long_window = 20
    df['short_mavg'] = df['close'].rolling(window=short_window, min_periods=short_window).mean()
    df['long_mavg'] = df['close'].rolling(window=long_window, min_periods=long_window).mean()
    mavg_crossover = df['short_mavg'] - df['long_mavg']

    # Integrate macroeconomic indicators (assuming we have an exogenous DataFrame `macro_df` with the same index as `df`)
    # For example, let's assume `macro_df` has a column 'inflation_rate'
    macro_df = pd.DataFrame({'inflation_rate': [0.01] * len(df)}, index=df.index)  # Placeholder for actual macro data
    df['inflation_rate'] = macro_df['inflation_rate']

    # Final alpha factor: combination of the ratio, combined factor, seasonality, and macroeconomic indicators
    final_factor = (factor1 + combined_factor) / 2 + mavg_crossover + df['inflation_rate']

    return final_factor
