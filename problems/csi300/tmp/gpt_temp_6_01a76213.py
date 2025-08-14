import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily Log Return
    df['log_return'] = np.log(df['close']).diff()
    
    # Exponential Moving Average of Log Returns
    lookback_period = 14
    df['price_momentum_score'] = df['log_return'].ewm(span=lookback_period).mean()
    
    # Calculate Daily Volume Change
    df['volume_change'] = df['volume'].pct_change()
    
    # Exponential Moving Average of Volume Changes
    df['volume_momentum_score'] = df['volume_change'].ewm(span=lookback_period).mean()
    
    # Integrate Sentiment Score (Assuming a 'sentiment' column exists in the DataFrame)
    df['sentiment_momentum_score'] = df['sentiment'].ewm(span=lookback_period).mean()
    
    # Combine Price, Volume, and Sentiment Scores
    df['combined_momentum_score'] = df['price_momentum_score'] * df['volume_momentum_score'] * df['sentiment_momentum_score']
    
    # Subtract Negative Impact
    df['daily_negative_returns'] = df['log_return'].apply(lambda x: x**2 if x < 0 else 0)
    df['downside_volatility'] = df['daily_negative_returns'].rolling(window=lookback_period).sum().pow(0.5)
    df['final_momentum_score'] = df['combined_momentum_score'] - df['downside_volatility']
    
    # Incorporate Macroeconomic Indicators (Assuming 'gdp_growth' and 'inflation_rate' columns exist)
    macro_weights = {'gdp_growth': 0.6, 'inflation_rate': 0.4}
    df['macro_average'] = (df['gdp_growth'] * macro_weights['gdp_growth']) + (df['inflation_rate'] * macro_weights['inflation_rate'])
    
    # Final Alpha Factor
    df['alpha_factor'] = df['final_momentum_score'] * df['macro_average']
    
    return df['alpha_factor']
