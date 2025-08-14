import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Return
    df['intraday_return'] = (df['high'] - df['low']) / df['low']
    
    # Momentum Confirmation
    df['momentum_confirmation_close'] = (df['close'].shift(-1) - df['close']) / df['close']
    df['momentum_confirmation_open'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['reversal_momentum_indicator'] = df['intraday_return'] * (df['momentum_confirmation_close'] + df['momentum_confirmation_open'])
    
    # Filter for Strong Signal Days based on Amount
    amount_threshold = df['amount'].quantile(0.75)
    df['reversal_momentum_filtered'] = df.apply(lambda row: row['reversal_momentum_indicator'] if row['amount'] > amount_threshold else 0, axis=1)
    
    # Calculate VWAP
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (df['typical_price'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Open-Close Spread Factor
    df['vwap_diff_open'] = df['vwap'] - df['open']
    df['vwap_diff_close'] = df['vwap'] - df['close']
    
    # Combine Reversal-Momentum Indicator and VWAP Differences
    df['weighted_intraday_return'] = df['intraday_return'] * df['vwap_diff_open']
    df['weighted_momentum_confirmation'] = df['momentum_confirmation_close'] * df['vwap_diff_close']
    df['combined_reversal_momentum_vwap'] = df['weighted_intraday_return'] + df['weighted_momentum_confirmation']
    
    # Calculate Raw Returns
    df['raw_returns'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Determine Volume Trend Direction
    df['volume_trend'] = df.apply(lambda row: 1 if row['volume'] > row['volume'].shift(1) else -1, axis=1)
    
    # Combine Return and Volume Indicators
    df['combined_indicator'] = df['raw_returns'] * df['volume_trend']
    
    # Sum Over a Window
    window_size = 5
    df['summed_indicator'] = df['combined_indicator'].rolling(window=window_size).sum()
    
    # Calculate Cumulative Momentum
    df['price_change'] = df['close'] - df['close'].shift(5)
    df['cumulative_momentum'] = df['price_change'].rolling(window=window_size).sum()
    
    # Calculate Volume Surge
    df['avg_volume'] = df['volume'].rolling(window=5).mean().shift(1)
    df['volume_ratio'] = df['volume'] / df['avg_volume']
    volume_surge_threshold = 1.5
    df['volume_surge'] = df.apply(lambda row: 1 if row['volume_ratio'] > volume_surge_threshold else 0, axis=1)
    
    # Calculate Price Surge
    price_surge_threshold = 1.2
    df['price_ratio'] = df['high'] / df['low']
    df['price_surge'] = df.apply(lambda row: 1 if row['price_ratio'] > price_surge_threshold else 0, axis=1)
    
    # Integrate Components
    df['integrated_indicators'] = df['summed_indicator'] * df['cumulative_momentum'] * df['volume_surge'] * df['price_surge']
    
    # Final Alpha Factor
    df['final_alpha_factor'] = df['integrated_indicators'] + df['combined_reversal_momentum_vwap']
    
    return df['final_alpha_factor']

# Example usage:
# df = pd.read_csv('your_data.csv')
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
