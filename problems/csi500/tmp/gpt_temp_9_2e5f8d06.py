import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame, macro_data: pd.DataFrame) -> pd.Series:
    # Adaptive momentum considering both short and long-term trends with exponential smoothing
    short_term_momentum = df['close'].pct_change(periods=5).ewm(span=5, adjust=False).mean()
    long_term_momentum = df['close'].pct_change(periods=30).ewm(span=30, adjust=False).mean()
    adaptive_momentum = 0.7 * short_term_momentum + 0.3 * long_term_momentum

    # Dynamic volatility using the standard deviation of logarithmic returns over the last 14 days with exponential smoothing
    log_returns = np.log(df['close'] / df['close'].shift(1))
    dynamic_volatility = log_returns.rolling(window=14).std().ewm(span=14, adjust=False).mean()

    # Normalized volume to account for varying liquidity conditions
    normalized_volume = (df['volume'] - df['volume'].mean()) / df['volume'].std()

    # Market sentiment using the ratio of high to low prices as a proxy
    market_sentiment = (df['high'] - df['low']) / df['close']

    # Incorporate macroeconomic indicators (e.g., interest rates, GDP growth)
    # Ensure the macro data is aligned with the stock data
    macro_data_aligned = macro_data.reindex(df.index, method='ffill')
    macro_effect = (macro_data_aligned['interest_rate'] * df['close'].pct_change(periods=1).rolling(window=30).mean() +
                    macro_data_aligned['gdp_growth'])

    # Sector-specific factors (e.g., sector performance)
    # For simplicity, we'll use a placeholder for these indicators
    sector_performance = 0.01  # Placeholder value, replace with actual sector-specific data
    sector_effect = sector_performance * (df['close'].pct_change(periods=1).rolling(window=30).mean())

    # Combine the factors into a single alpha factor
    factor_values = (
        0.4 * adaptive_momentum
        - 0.2 * dynamic_volatility
        + 0.3 * normalized_volume
        + 0.1 * market_sentiment
        + 0.1 * macro_effect
        + 0.1 * sector_effect
    )

    # Use machine learning for dynamic factor combination
    from sklearn.linear_model import ElasticNet
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error

    # Create a DataFrame with the features
    features = pd.DataFrame({
        'adaptive_momentum': adaptive_momentum,
        'dynamic_volatility': dynamic_volatility,
        'normalized_volume': normalized_volume,
        'market_sentiment': market_sentiment,
        'macro_effect': macro_effect,
        'sector_effect': sector_effect
    }).dropna()

    # Target variable (future returns)
    target = df['close'].pct_change(periods=1).shift(-1).loc[features.index]

    # Split the data into training and testing sets
    tscv = TimeSeriesSplit(n_splits=3)
    for train_index, test_index in tscv.split(features):
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]

        # Train an elastic net model
        model = ElasticNet(alpha=0.1, l1_ratio=0.5)
        model.fit(X_train, y_train)

        # Predict future returns
        predictions = model.predict(X_test)

        # Adjust the factor values based on the model's predictions
        factor_values.loc[X_test.index] = predictions

    return factor_values
