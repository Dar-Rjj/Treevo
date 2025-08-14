import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def heuristics_v2(df):
    # Calculate Short-Term Price Momentum
    short_term_momentum = df['close'] - df['close'].rolling(window=10).mean()
    
    # Calculate Medium-Term Price Momentum
    medium_term_momentum = df['close'] - df['close'].rolling(window=30).mean()
    
    # Calculate Long-Term Price Momentum
    long_term_momentum = df['close'] - df['close'].rolling(window=50).mean()
    
    # Combine Multi-Period Momenta
    combined_momentum = short_term_momentum + medium_term_momentum + long_term_momentum
    
    # Calculate Volume-Weighted Average Return
    daily_returns = (df['close'] - df['open']) / df['open']
    volume_weighted_returns = (daily_returns * df['volume']).sum() / df['volume'].sum()
    
    # Adjust Combined Momentum by Volume-Weighted Average Return
    adjusted_combined_momentum = combined_momentum * volume_weighted_returns
    
    # Assess Trend Following Potential
    fifty_day_ma = df['close'].rolling(window=50).mean()
    trend_weight = np.where(fifty_day_ma > df['close'], 1, 0.5)
    trend_component = trend_weight * (df['close'] - fifty_day_ma)
    
    # Determine Initial Factor Value
    initial_factor_value = adjusted_combined_momentum + trend_component
    
    # Calculate Short-Term Volatility
    short_term_volatility = (df['high'] - df['low']).rolling(window=10).mean()
    
    # Calculate Medium-Term Volatility
    medium_term_volatility = (df['high'] - df['low']).rolling(window=30).mean()
    
    # Calculate Long-Term Volatility
    long_term_volatility = (df['high'] - df['low']).rolling(window=50).mean()
    
    # Combine Multi-Period Volatilities
    combined_volatility = short_term_volatility + medium_term_volatility + long_term_volatility
    
    # Adjust Initial Factor Value by Combined Volatility
    adjusted_factor_value = initial_factor_value / combined_volatility
    
    # Integrate Macro Trends
    sp500_trend = np.where(df['sp500_index'] > df['sp500_index'].shift(1), 1, -1)
    macro_adjusted_factor_value = adjusted_factor_value * sp500_trend
    
    # Enhance with Machine Learning
    X = df[['short_term_momentum', 'medium_term_momentum', 'long_term_momentum', 
            'combined_momentum', 'volume_weighted_returns', 'trend_component',
            'combined_volatility', 'sp500_trend']].dropna()
    y = df['future_returns'].shift(-1).dropna()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predicted_adjustments = model.predict(X.dropna())
    final_factor_value = macro_adjusted_factor_value * predicted_adjustments
    
    return final_factor_value
