import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def heuristics_v2(df, macroeconomic_data):
    # Calculate Intraday Return
    intraday_return = df['close'] - df['open']
    
    # Calculate Intraday High-Low Range
    high_low_range = df['high'] - df['low']
    
    # Combine Intraday Return and High-Low Range
    combined_factor = intraday_return * high_low_range
    smoothed_factor = combined_factor.ewm(span=14, adjust=False).mean()
    
    # Apply Volume Weighting
    volume_weighted_factor = smoothed_factor * df['volume']
    
    # Incorporate Previous Day's Closing Gap
    closing_gap = df['open'].shift(-1) - df['close'].shift(1)
    volume_weighted_closing_gap = volume_weighted_factor + closing_gap
    
    # Integrate Long-Term Momentum
    long_term_return = df['close'] - df['close'].shift(50)
    normalized_long_term_return = long_term_return / high_low_range
    
    # Include Enhanced Dynamic Volatility Component
    rolling_std = df['close'].pct_change().rolling(window=20).std()
    atr = (df['high'] - df['low']).rolling(window=14).mean()
    combined_volatility = (rolling_std + atr) / 2
    volume_adjusted_volatility = combined_volatility * df['volume']
    
    # Incorporate Macroeconomic Indicators
    macro_scaled = (macroeconomic_data - macroeconomic_data.mean()) / macroeconomic_data.std()
    
    # Final Factor Calculation
    final_factor = (volume_weighted_closing_gap + 
                    normalized_long_term_return + 
                    volume_adjusted_volatility + 
                    macro_scaled)
    
    # Apply Machine Learning for Non-Linear Transformation
    X_train = final_factor.dropna().values.reshape(-1, 1)
    y_train = df['close'].pct_change().dropna()[X_train.index].values
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predicted_factors = model.predict(final_factor.values.reshape(-1, 1))
    
    return pd.Series(predicted_factors, index=df.index)

# Example usage:
# df = pd.read_csv('your_data.csv', parse_dates=['date'], index_col='date')
# macroeconomic_data = pd.read_csv('macroeconomic_data.csv', parse_dates=['date'], index_col='date')['GDP_growth']
# alpha_factor = heuristics_v2(df, macroeconomic_data)
