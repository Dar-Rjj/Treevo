import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    # Calculate the positive and negative part of the amount over volume
    df['positive_amount_vol'] = (df['amount'] / df['volume']).clip(lower=0)
    df['negative_amount_vol'] = (df['amount'] / df['volume']).clip(upper=0)

    # Sum of positive and absolute negative parts
    pos_sum = df['positive_amount_vol'].rolling(window=5).sum()
    neg_sum_abs = df['negative_amount_vol'].abs().rolling(window=5).sum()

    # Factor: ratio of positive sum to absolute negative sum
    sentiment_factor = pos_sum / (neg_sum_abs + 1e-7)

    # Calculate the volatility using the close price
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    volatility = df['log_returns'].rolling(window=20).std() * np.sqrt(252)

    # Calculate the VWAP
    df['vwap'] = (df['amount'] / df['volume']).rolling(window=20).mean()

    # Exponential smoothing on the VWAP
    df['vwap_smoothed'] = df['vwap'].ewm(span=20, adjust=False).mean()

    # Momentum factor
    momentum = df['close'].pct_change(periods=20)

    # Mean reversion factor
    mean_reversion = -df['close'].pct_change(periods=5)

    # Trend factor using exponential moving average
    ema_10 = df['close'].ewm(span=10, adjust=False).mean()
    ema_30 = df['close'].ewm(span=30, adjust=False).mean()
    trend_factor = (ema_10 - ema_30) / df['close']

    # Market depth factor
    market_depth = (df['high'] - df['low']) / df['close']

    # Adaptive window for volatility
    adaptive_volatility = df['log_returns'].expanding(min_periods=20).std() * np.sqrt(252)

    # Combine factors
    factors_df = pd.DataFrame({
        'sentiment': sentiment_factor,
        'volatility_inv': 1 / (adaptive_volatility + 1e-7),
        'vwap_ratio': df['vwap'] / df['vwap_smoothed'],
        'momentum': momentum,
        'mean_reversion': mean_reversion,
        'trend': trend_factor,
        'market_depth': market_depth
    })

    # Drop NaN values
    factors_df = factors_df.dropna()

    # Use linear regression to dynamically weight the factors
    X = factors_df.values
    y = df['close'].pct_change().dropna()[-len(factors_df):].values
    model = LinearRegression().fit(X, y)
    weights = model.coef_

    # Apply the dynamic weights
    alpha_factor = (factors_df * weights).sum(axis=1)

    return alpha_factor
