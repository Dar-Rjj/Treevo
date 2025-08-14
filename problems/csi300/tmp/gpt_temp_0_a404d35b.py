import pandas as pd
def heuristics_v2(df: pd.DataFrame, macroeconomic_df: pd.DataFrame) -> pd.Series:
    # Dynamic window sizes based on market conditions
    short_window = 50
    long_window = 100
    liquidity_window = 60
    volatility_window = 30
    mfi_window = 20
    sentiment_window = 10

    # Adaptive Momentum - 50 to 100 period return based on the current market trend
    short_momentum = df['close'].pct_change(short_window)
    long_momentum = df['close'].pct_change(long_window)
    momentum = (short_momentum + long_momentum) / 2

    # Liquidity - Calculate the average volume over a 60 day period to smooth out short-term fluctuations
    liquidity = df['volume'].rolling(window=liquidity_window).mean()

    # Volatility - Calculate the rolling standard deviation of daily returns over a 30 day period for a more stable measure
    daily_returns = df['close'].pct_change()
    volatility = daily_returns.rolling(window=volatility_window).std()

    # True Range (TR) calculation for volatility
    prev_close = df['close'].shift(1)
    tr = (df['high'] - df['low']).abs()
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (prev_close - df['low']).abs()
    avg_true_range = (tr + tr2 + tr3).rolling(window=volatility_window).mean()  # Use rolling mean for TR

    # Money Flow Index (MFI) with a 20-day period
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    raw_money_flow = typical_price * df['volume']
    positive_money_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=mfi_window).sum()
    negative_money_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=mfi_window).sum()
    mfi = 100 - (100 / (1 + positive_money_flow / (negative_money_flow + 1e-7)))

    # Market Sentiment - Calculate the ratio of high to low prices, using a 10-day moving average for smoothing
    sentiment = (df['high'] / df['low']).rolling(window=sentiment_window).mean()

    # Price-Volume Interaction
    price_volume_interaction = df['close'] * df['volume']

    # Incorporate macroeconomic indicators
    macroeconomic_influence = macroeconomic_df['indicator'].ewm(span=60, adjust=False).mean()

    # Composite alpha factor
    alpha_factor = (momentum * liquidity / (volatility + 1e-7)) * (mfi / 100) * (avg_true_range / df['close']) * (sentiment - 1) * price_volume_interaction * macroeconomic_influence

    # Machine learning for adaptive factor weighting
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Prepare features and target
    X = df[['close', 'volume', 'high', 'low']].pct_change().dropna()
    y = df['close'].shift(-1).pct_change().dropna()
    X, y = X.align(y, join='inner')

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Predict and weight factors
    predictions = model.predict(scaler.transform(df[['close', 'volume', 'high', 'low']].pct_change().dropna()))
    factor_weights = predictions[-len(alpha_factor):]

    # Apply weights to alpha factor
    weighted_alpha_factor = alpha_factor * factor_weights

    return weighted_alpha_factor
