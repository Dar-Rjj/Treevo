import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from arch import arch_model
from sklearn.ensemble import RandomForestRegressor

def heuristics_v2(df, sector_df, window=14, long_term_window=50, garch_window=20):
    # Calculate Intraday Return
    df['intraday_return'] = df['close'] - df['open']
    
    # Calculate Intraday High-Low Range
    df['high_low_range'] = df['high'] - df['low']
    
    # Combine Intraday Return and High-Low Range
    df['combined_factor'] = df['intraday_return'] * df['high_low_range']
    
    # Smooth using Exponential Moving Average (EMA)
    df['ema_smoothed'] = df['combined_factor'].ewm(span=window).mean()
    
    # Apply Volume Weighting
    df['volume_weighted'] = df['ema_smoothed'] * df['volume']
    
    # Incorporate Previous Day's Closing Gap
    df['previous_day_gap'] = df['open'] - df['close'].shift(1)
    df['volume_weighted_with_gap'] = df['volume_weighted'] + df['previous_day_gap']
    
    # Integrate Long-Term Momentum
    df['long_term_return'] = df['close'] - df['close'].shift(long_term_window)
    df['normalized_long_term_return'] = df['long_term_return'] / df['high_low_range']
    
    # Include Sector-Specific Momentum
    # Assuming sector_df is a DataFrame with the same index and a column 'sector_close' for the sector's close price
    sector_df['sector_long_term_return'] = sector_df['sector_close'] - sector_df['sector_close'].shift(long_term_window)
    sector_df['normalized_sector_return'] = sector_df['sector_long_term_return'] / (df['high'] - df['low'])
    df['normalized_sector_return'] = sector_df['normalized_sector_return']
    
    # Include Adaptive Volatility Component
    df['intraday_returns_rolling_std'] = df['intraday_return'].rolling(window=garch_window).std()
    
    # Calculate GARCH(1,1) Volatility
    garch_model = arch_model(df['intraday_return'].dropna(), vol='Garch', p=1, q=1, dist='Normal')
    garch_results = garch_model.fit(disp='off')
    df['garch_volatility'] = garch_results.conditional_volatility
    
    # Combine Rolling Standard Deviation and GARCH Volatility
    df['combined_volatility'] = (df['intraday_returns_rolling_std'] + df['garch_volatility']) / 2
    df['volume_adjusted_volatility'] = df['combined_volatility'] * df['volume']
    
    # Final Factor Calculation
    df['final_factor'] = (
        df['volume_weighted_with_gap'] +
        df['normalized_long_term_return'] +
        df['normalized_sector_return'] +
        df['volume_adjusted_volatility']
    )
    
    # Apply Non-Linear Transformation using Machine Learning
    X = df[['volume_weighted_with_gap', 'normalized_long_term_return', 'normalized_sector_return', 'volume_adjusted_volatility']].dropna()
    y = df['return'].shift(-1).loc[X.index]  # Target variable is next day's return
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    df['ml_transformed_factor'] = model.predict(X)
    
    return df['ml_transformed_factor']

# Example usage:
# df = pd.read_csv('stock_data.csv', index_col='date', parse_dates=True)
# sector_df = pd.read_csv('sector_data.csv', index_col='date', parse_dates=True)
# alpha_factor = heuristics_v2(df, sector_df)
