import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def heuristics_v2(df):
    # Momentum and Trend Indicators
    df['raw_price_momentum'] = df['close'].diff(20)
    df['volume_weighted_momentum'] = (df['close'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    df['multi_period_momentum'] = df['close'].pct_change(10) + df['close'].pct_change(20) + df['close'].pct_change(50) + df['close'].pct_change(100)
    df['ma_50'] = df['close'].rolling(window=50).mean()
    df['ma_200'] = df['close'].rolling(window=200).mean()
    df['moving_average_crossover'] = df['ma_50'] - df['ma_200']
    
    # Volatility and Risk Adjustments
    df['daily_returns'] = df['close'].pct_change()
    df['historical_volatility'] = df['daily_returns'].rolling(window=20).std()
    df['adaptive_volatility'] = df['daily_returns'].ewm(span=20).std()
    
    # Liquidity and Market Depth
    df['average_volume_20'] = df['volume'].rolling(window=20).mean()
    df['trading_volume_ratio'] = df['volume'] / df['average_volume_20']
    df['dollar_volume'] = df['close'] * df['volume']
    df['volume_weighted_price'] = (df['close'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    
    # Sector Rotation and Diversification
    # Assuming sector data is available in the DataFrame
    if 'sector' in df.columns:
        df['sector_relative_strength'] = df.groupby('sector')['close'].pct_change(30)
        df['sector_momentum'] = df.groupby('sector')['close'].pct_change(30).mean()
    
    # Machine Learning for Dynamic Weighting
    features = ['raw_price_momentum', 'volume_weighted_momentum', 'multi_period_momentum', 'moving_average_crossover', 
                'historical_volatility', 'adaptive_volatility', 'trading_volume_ratio', 'dollar_volume', 'volume_weighted_price']
    X = df[features].dropna()
    y = df['close'].shift(-1).loc[X.index]  # Predict next day's close price
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    df['predicted_return'] = model.predict(df[features].fillna(0))
    
    # Macroeconomic Indicators
    # Assuming GDP, inflation, and interest rates are available in the DataFrame
    if 'gdp_growth_rate' in df.columns:
        df['gdp_growth_rate_lag'] = df['gdp_growth_rate'].shift(1)
    if 'inflation_rate' in df.columns:
        df['inflation_rate_lag'] = df['inflation_rate'].shift(1)
    if 'interest_rate' in df.columns:
        df['interest_rate_lag'] = df['interest_rate'].shift(1)
    
    # Combine Factors into a Composite Alpha Signal
    df['composite_score'] = df['raw_price_momentum'] + df['volume_weighted_momentum'] + df['multi_period_momentum'] + \
                            df['moving_average_crossover'] - df['historical_volatility'] - df['adaptive_volatility'] + \
                            df['trading_volume_ratio'] + df['dollar_volume'] + df['volume_weighted_price'] + df['predicted_return']
    
    if 'gdp_growth_rate_lag' in df.columns and 'inflation_rate_lag' in df.columns and 'interest_rate_lag' in df.columns:
        df['composite_score'] += df['gdp_growth_rate_lag'] - df['inflation_rate_lag'] + df['interest_rate_lag']
    
    return df['composite_score'].fillna(0)
