import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Calculate daily price change
    df['price_change'] = df['close'] - df['open']
    
    # Identify trend direction
    df['trend_direction'] = (df['close'] > df['open']).astype(int)
    
    # Calculate high-low spread
    df['high_low_spread'] = df['high'] - df['low']
    
    # Calculate open-close ratio
    df['open_close_ratio'] = df['open'] / df['close']
    
    # Calculate day-to-day return
    df['day_to_day_return'] = (df['close'].shift(1) - df['close']) / df['close'].shift(1)
    
    # Compare volume to previous day's volume
    df['volume_change'] = df['volume'] - df['volume'].shift(1)
    
    # Calculate amount-volume ratio
    df['amount_volume_ratio'] = df['amount'] / df['volume']
    
    # Check for significant volume changes
    df['significant_volume_increase'] = (df['volume'] > 1.5 * df['volume'].shift(1)).astype(int)
    df['significant_volume_decrease'] = (df['volume'] < 0.5 * df['volume'].shift(1)).astype(int)
    
    # Calculate price volatility
    df['price_volatility'] = df['close'].rolling(window=20).std()
    
    # Measure price momentum
    def calculate_momentum(close_prices):
        x = np.arange(len(close_prices))
        slope, _, _, _, _ = linregress(x, close_prices)
        return slope
    
    df['price_momentum'] = df['close'].rolling(window=5).apply(calculate_momentum, raw=False)
    
    # Calculate average true range
    df['true_range'] = df[['high', 'close']].max(axis=1) - df[['low', 'close']].min(axis=1)
    df['average_true_range'] = df['true_range'].rolling(window=14).mean()
    
    # Analyze volume-weighted average price
    df['vwap'] = (df['close'] * df['volume']).rolling(window=10).sum() / df['volume'].rolling(window=10).sum()
    
    # Detect breakout patterns
    df['potential_breakout'] = (df['close'] > df['high'].rolling(window=20).max()).astype(int)
    df['potential_breakdown'] = (df['close'] < df['low'].rolling(window=20).min()).astype(int)
    
    # Evaluate volume spikes
    df['volume_spike'] = (df['volume'] > 2 * df['volume'].rolling(window=30).mean()).astype(int)
    
    # Calculate on-balance volume
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    
    # Create a composite score
    df['composite_score'] = (
        0.2 * df['price_change'] +
        0.1 * df['trend_direction'] +
        0.1 * df['high_low_spread'] +
        0.1 * df['open_close_ratio'] +
        0.1 * df['day_to_day_return'] +
        0.1 * df['volume_change'] +
        0.1 * df['amount_volume_ratio'] +
        0.1 * df['significant_volume_increase'] +
        0.1 * df['significant_volume_decrease'] +
        0.1 * df['price_volatility'] +
        0.1 * df['price_momentum'] +
        0.1 * df['average_true_range'] +
        0.1 * df['vwap'] +
        0.1 * df['potential_breakout'] +
        0.1 * df['potential_breakdown'] +
        0.1 * df['volume_spike'] +
        0.1 * df['obv']
    )
    
    # Return the composite score as the alpha factor
    return df['composite_score']
