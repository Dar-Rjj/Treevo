import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Compute Price Change
    df['price_change'] = df['close'].diff()
    
    # Incorporate Volume Impact Factor
    df['momentum_contribution'] = df['volume'] * df['price_change'].abs()
    
    # Integrate Historical Momentum Contributions
    df['5_day_momentum_sum'] = df['momentum_contribution'].rolling(window=5).sum()
    
    # Apply Positive Slope Emphasis
    df['price_change_5d'] = df['price_change'].rolling(window=5).apply(lambda x: x.max() - x.min(), raw=True)
    df['positive_slope'] = df['price_change_5d'].apply(lambda x: x if x > 0 else 0)
    
    # Accumulate Weighted Contributions
    df['accumulated_momentum'] = df['5_day_momentum_sum'] * (df['positive_slope'] > 0)
    
    # Adjust for Market Sentiment
    df['volatility'] = (df['high'] - df['low']) / df['close']
    df['5_day_volatility_avg'] = df['volatility'].rolling(window=5).mean()
    
    # Conditional Adjustment to AVMI
    df['avmi_adjusted'] = df['accumulated_momentum'].apply(
        lambda x: x + (x - df['5_day_volatility_avg']) if x > df['5_day_volatility_avg'] else x - (df['5_day_volatility_avg'] - x)
    )
    
    # Calculate Volume Trend Indicator
    df['daily_volume_change'] = df['volume'].diff()
    df['volume_trend'] = df['daily_volume_change'].rolling(window=5).sum()
    
    # Combine Volume and Price Momentum
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['combined_momentum'] = df['log_return'] * df['volume_trend']
    
    # Finalize and Output Combined Alpha Factor
    df['alpha_factor'] = df['avmi_adjusted'] * df['combined_momentum']
    
    return df['alpha_factor'].dropna()
