import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate daily returns
    df['daily_return'] = df['close'].pct_change()
    
    # Calculate adaptive 20-day volatility
    df['volatility'] = df['daily_return'].rolling(window=20, min_periods=1).std()
    
    # Calculate adaptive 20-day momentum
    df['momentum'] = df['close'].pct_change(periods=20)
    
    # Calculate average volume over an adaptive 20-day window
    df['avg_volume'] = df['volume'].rolling(window=20, min_periods=1).mean()
    
    # Calculate the longer-term trend (50-day moving average)
    df['long_term_trend'] = df['close'].rolling(window=50, min_periods=1).mean()
    
    # Calculate the distance of the current close price from the long-term trend
    df['trend_distance'] = (df['close'] - df['long_term_trend']) / df['long_term_trend']
    
    # Incorporate market sentiment by considering the ratio of the daily return to the volatility
    df['sentiment'] = df['daily_return'] / (df['volatility'] + 1e-7)
    
    # Weighted factor combining momentum, inverse volatility, trend distance, and market sentiment
    alpha_factor = (df['momentum'] / (df['volatility'] + 1e-7)) * (df['avg_volume'] / (df['volume'] + 1e-7)) * df['trend_distance'] * df['sentiment']
    
    # Integrate sector performance (assuming a 'sector' column in the DataFrame)
    if 'sector' in df.columns:
        sector_returns = df.groupby('sector')['daily_return'].transform(lambda x: x.rolling(window=20, min_periods=1).mean())
        sector_volatility = df.groupby('sector')['daily_return'].transform(lambda x: x.rolling(window=20, min_periods=1).std())
        sector_sentiment = sector_returns / (sector_volatility + 1e-7)
        alpha_factor *= sector_sentiment
    
    # Use machine learning to find optimal weights for the factors
    # Here, we use a simple linear regression model as an example
    from sklearn.linear_model import LinearRegression
    X = df[['momentum', 'volatility', 'avg_volume', 'trend_distance', 'sentiment']].dropna()
    y = df['daily_return'].shift(-1).dropna().loc[X.index]  # Predicting next day's return
    model = LinearRegression().fit(X, y)
    df.loc[X.index, 'alpha_factor_ml'] = X @ model.coef_
    
    # Combine the machine learning-based factor with the original alpha factor
    alpha_factor = alpha_factor * df['alpha_factor_ml']
    
    return alpha_factor
