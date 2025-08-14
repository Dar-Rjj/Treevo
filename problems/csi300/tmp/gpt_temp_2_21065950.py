import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily Log Return
    df['log_return'] = np.log(df['close']).diff()
    
    # Calculate Price Momentum Score using EMA of log returns
    lookback_period = 14
    df['price_momentum'] = df['log_return'].ewm(span=lookback_period, adjust=False).mean()
    
    # Calculate Daily Volume Change
    df['volume_change'] = df['volume'].pct_change()
    
    # Calculate Volume Momentum Score using EMA of volume changes
    df['volume_momentum'] = df['volume_change'].ewm(span=lookback_period, adjust=False).mean()
    
    # Combine Price and Volume Momentum Scores
    df['combined_momentum'] = df['price_momentum'] * df['volume_momentum']
    
    # Calculate Aroon Oscillator for Market Sentiment
    aroon_up = 100 * (lookback_period - df['high'].rolling(window=lookback_period).apply(lambda x: list(x).index(max(x))))
    aroon_down = 100 * (lookback_period - df['low'].rolling(window=lookback_period).apply(lambda x: list(x).index(min(x))))
    df['aroon_oscillator'] = aroon_up - aroon_down
    
    # Adjust Combined Momentum Score with Market Sentiment
    df['adjusted_momentum'] = df['combined_momentum'] * df['aroon_oscillator']
    
    # Calculate Downside Volatility
    daily_neg_returns = df['log_return'].where(df['log_return'] < 0, 0)
    df['downside_volatility'] = np.sqrt(np.sum(daily_neg_returns**2))
    
    # Subtract Downside Volatility from Adjusted Momentum Score
    df['final_alpha_factor'] = df['adjusted_momentum'] - df['downside_volatility']
    
    return df['final_alpha_factor']

# Example usage:
# df = pd.read_csv('your_data.csv', index_col='date')
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
