import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import zscore

def heuristics_v2(df):
    # Calculate daily returns
    df['returns'] = df['close'].pct_change()
    
    # Calculate Volume-Weighted Average Price (VWAP)
    df['vwap'] = (df['volume'] * (df['high'] + df['low']) / 2).cumsum() / df['volume'].cumsum()
    df['vwap_returns'] = df['vwap'].pct_change()
    
    # Calculate volume-weighted momentum
    df['vwap_momentum'] = df['vwap_returns'].rolling(window=10).mean()
    
    # Calculate volume-weighted volatility
    df['vwap_volatility'] = df['vwap_returns'].rolling(window=10).std()
    
    # Adaptive EMA window for price based on price volatility
    def adaptive_ema(data, window):
        alpha = 2 / (window + 1)
        ema = data.ewm(alpha=alpha, adjust=False).mean()
        return ema
    
    price_volatility = df['returns'].rolling(window=10).std()
    adaptive_window_price = 10 + (price_volatility * 10).astype(int)
    df['adaptive_ema_price'] = adaptive_ema(df['close'], adaptive_window_price)
    
    # Adaptive EMA window for volume based on volume changes
    volume_changes = df['volume'].pct_change().abs()
    adaptive_window_volume = 10 + (volume_changes * 10).astype(int)
    df['adaptive_ema_volume'] = adaptive_ema(df['volume'], adaptive_window_volume)
    
    # Combined EMA factor
    df['combined_ema_factor'] = (df['adaptive_ema_price'] + df['adaptive_ema_volume']) / 2
    df['combined_ema_factor'] = df['combined_ema_factor'].ewm(span=10, adjust=False).mean()
    
    # Dynamic factor weighting
    short_term_factors = (df['returns'] + df['vwap_momentum']).rolling(window=5).mean()
    long_term_factors = (df['returns'] + df['vwap_momentum']).rolling(window=20).mean()
    
    # Balanced factor model
    df['balanced_factor'] = (short_term_factors + long_term_factors) / 2
    df['balanced_factor'] = df['balanced_factor'].ewm(span=10, adjust=False).mean()
    
    # Final alpha factor
    df['alpha_factor'] = df['vwap_volatility'] + df['combined_ema_factor'] + df['balanced_factor']
    df['alpha_factor'] = zscore(df['alpha_factor'])
    
    return df['alpha_factor']

# Example usage
# df = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
