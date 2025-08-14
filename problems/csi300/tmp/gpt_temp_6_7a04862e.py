import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame, sector_trends: pd.Series, macro_indicators: pd.DataFrame) -> pd.Series:
    # Calculate the 5-day and 20-day simple moving average of closing prices to capture momentum
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()

    # Momentum factor: difference between 5-day and 20-day moving averages
    df['momentum_factor'] = df['SMA_5'] - df['SMA_20']

    # Calculate the 20-day standard deviation of closing prices to capture volatility
    df['volatility_factor'] = df['close'].rolling(window=20).std()

    # Calculate the 5-day rolling sum of volume to capture liquidity
    df['liquidity_factor'] = df['volume'].rolling(window=5).sum()

    # Calculate relative strength as the ratio of current close to 20-day SMA
    df['relative_strength'] = df['close'] / df['SMA_20']

    # Calculate market breadth: difference between 5-day and 20-day moving averages of (high - low)
    df['range_5'] = (df['high'] - df['low']).rolling(window=5).mean()
    df['range_20'] = (df['high'] - df['low']).rolling(window=20).mean()
    df['market_breadth'] = df['range_5'] - df['range_20']

    # Incorporate sector-specific trends
    df['sector_trend'] = sector_trends.reindex(df.index, method='ffill')

    # Incorporate macroeconomic indicators
    for col in macro_indicators.columns:
        df[col] = macro_indicators[col].reindex(df.index, method='ffill')

    # Use machine learning for dynamic factor weighting
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np

    # Prepare the features and target
    X = df[['momentum_factor', 'volatility_factor', 'liquidity_factor', 'relative_strength', 'market_breadth', 'sector_trend'] + list(macro_indicators.columns)].dropna()
    y = df['close'].pct_change().shift(-1).dropna()

    # Ensure the indices match
    y = y.loc[X.index]

    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Predict the weights
    predicted_weights = model.predict(X)

    # Apply the predicted weights to the factors
    df['predicted_weight'] = np.nan
    df.loc[X.index, 'predicted_weight'] = predicted_weights

    # Combine the factors with the predicted weights
    heuristic_factor = (df['momentum_factor'] * df['predicted_weight']) + \
                       (df['liquidity_factor'] * df['predicted_weight']) + \
                       (df['relative_strength'] * df['predicted_weight']) + \
                       (df['market_breadth'] * df['predicted_weight'])

    return heuristic_factor
