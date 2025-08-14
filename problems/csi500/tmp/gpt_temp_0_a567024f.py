import pandas as pd
import pandas as pd

def heuristics_v2(df, n=10):
    # Calculate Volume Dynamics
    df['V_sum'] = df['volume'].rolling(window=n).sum()
    df['V_change'] = df['V_sum'] - df['V_sum'].shift(n)
    
    # Evaluate High-Low Price Volatility and ATR
    df['H/L'] = df['high'] / df['low']
    df['ATR'] = df[['high' - 'low', (df['high'] - df['close'].shift(1)).abs(), (df['low'] - df['close'].shift(1)).abs()]].max(axis=1).rolling(window=n).mean()
    
    # Quantify Price Volatility
    df['price_volatility'] = df['close'].rolling(window=n).std()
    
    # Measure Volume Trend and Momentum
    df['volume_MA'] = df['volume'].rolling(window=n).mean()
    df['volume_trend'] = (df['volume'] > df['volume_MA']).astype(int) * 2 - 1  # +1 for increasing, -1 for decreasing
    df['volume_momentum'] = df['V_sum'] - df['V_sum'].shift(1)
    
    # Adjusted Volume and Price Range Factors
    df['adjusted_volume_momentum'] = df['V_change'] * df['H/L']
    df['inverted_volume_momentum'] = -1 * df['V_change']
    df['volume_adjusted_high_low_range'] = df['volume'] * df['H/L']
    
    # Weighted Combined Alpha Factor
    df['weighted_adjusted_volume_momentum'] = 0.4 * df['adjusted_volume_momentum']
    df['weighted_inverted_volume_momentum'] = 0.3 * df['inverted_volume_momentum']
    df['weighted_volume_adjusted_high_low_range'] = 0.2 * df['volume_adjusted_high_low_range']
    df['combined_alpha'] = (df['weighted_adjusted_volume_momentum'] + 
                            df['weighted_inverted_volume_momentum'] + 
                            0.3 * df['price_volatility'] + 
                            df['weighted_volume_adjusted_high_low_range'])
    
    # Integrate Open-Close Price Dynamics
    df['open_close_change'] = df['close'] - df['open']
    df['cumulative_open_close_change'] = df['open_close_change'].rolling(window=n).sum()
    
    # Incorporate 30-Day Price Momentum
    df['30_day_momentum'] = (df['close'] - df['close'].shift(30)) / df['close'].shift(30)
    
    # Identify Volume Shock
    df['avg_volume_30_days'] = df['volume'].rolling(window=30).mean().shift(1)
    df['volume_shock'] = (df['volume'] > 2 * df['avg_volume_30_days']).astype(int)
    
    # Combine Momentum and Volume Shock
    df['combined_momentum_and_shock'] = df['30_day_momentum'] * df['volume_shock']
    
    # Generate Enhanced Composite Alpha Factor
    df['adjusted_high_low_price_ratio'] = df['H/L'] * df['volume_trend']
    df['enhanced_composite_alpha_factor'] = (df['combined_alpha'] + 
                                            df['combined_momentum_and_shock'] + 
                                            df['cumulative_open_close_change'] * 
                                            (df['open_close_change'] > 0).astype(int) * 2 - 1)
    
    return df['enhanced_composite_alpha_factor']

# Example usage:
# df = pd.DataFrame(...)  # Load your DataFrame here
# alpha_factor = heuristics_v2(df)
