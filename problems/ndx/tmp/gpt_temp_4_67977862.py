import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, N=10, ATR_window=10, volume_window=20, percentile_threshold=75):
    # Calculate Intraday Range
    df['intraday_range'] = df['high'] - df['low']
    
    # Compute Price Change
    df['price_change'] = df['close'] - df['close'].shift(1)
    
    # Detect Significant Volume Increase
    df['average_volume'] = df['volume'].rolling(window=N).mean()
    df['volume_spike'] = df['volume'] > 2 * df['average_volume']
    
    # Normalize Price Change by Intraday Volatility
    df['adjusted_price_change'] = df['price_change'] / df['intraday_range']
    
    # Apply Volume-Weighted Adjustment
    df['weighted_adjusted_price_change'] = np.where(
        df['volume_spike'],
        df['volume'] * (df['adjusted_price_change'] * 2),
        df['volume'] * df['adjusted_price_change']
    )
    
    # Accumulate Momentum Score
    df['momentum_score'] = df['weighted_adjusted_price_change'].rolling(window=N).sum()
    
    # Calculate Rate of Change (ROC) for Closing Price
    df['roc'] = (df['close'] - df['close'].shift(14)) / df['close'].shift(14)
    
    # Incorporate Average True Range (ATR) for Short-Term Volatility
    df['true_range'] = df[['high', 'low', 'close']].apply(lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close'].shift(1)), abs(x['low'] - x['close'].shift(1))), axis=1)
    df['atr'] = df['true_range'].rolling(window=ATR_window).mean()
    
    # Integrate Trading Volume as a Measure of Market Interest
    df['average_volume_20'] = df['volume'].rolling(window=volume_window).mean()
    df['volume_indicator'] = df['volume'] / df['average_volume_20']
    
    # Combine the Factors into a Composite Alpha Factor
    df['composite_alpha_factor'] = (df['momentum_score'] + df['roc'] + df['atr'] + df['volume_indicator']) / 4
    
    # Apply a Threshold to Filter Out Signals
    threshold = np.percentile(df['composite_alpha_factor'].dropna(), percentile_threshold)
    df['alpha_signal'] = np.where(df['composite_alpha_factor'] > threshold, df['composite_alpha_factor'], np.nan)
    
    return df['alpha_signal']

# Example usage:
# df = pd.read_csv('your_data.csv', parse_dates=['date'], index_col='date')
# alpha_signal = heuristics_v2(df)
