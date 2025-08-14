import pandas as pd
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate price change and volume change
    df['price_change'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()

    # Adaptive windows for volatility based on the rolling standard deviation of the close price
    df['volatility_5'] = df['close'].rolling(window=5).std()
    df['volatility_20'] = df['close'].rolling(window=20).std()
    df['volatility_ratio'] = df['volatility_5'] / (df['volatility_20'] + 1e-7)

    # Momentum indicators
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['trend'] = df['sma_5'] / (df['sma_20'] + 1e-7)

    # Leverage as a weighted sum of price change, volume change, and volatility ratio
    leverage = (df['price_change'] + 2 * df['volume_change'] + 3 * df['volatility_ratio']) / 6

    # Combine leverage and trend to form an initial alpha factor
    df['alpha_factor_initial'] = leverage * df['trend']

    # Use machine learning to dynamically weight the factors
    X = df[['price_change', 'volume_change', 'volatility_ratio', 'trend']]
    y = df['alpha_factor_initial']

    # Train a RandomForestRegressor to predict the alpha factor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X.dropna(), y.dropna())

    # Predict the alpha factor using the trained model
    df['alpha_factor_ml'] = model.predict(X)

    return df['alpha_factor_ml']
