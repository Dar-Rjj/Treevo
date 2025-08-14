import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Price Movement
    df['high_low_range'] = df['high'] - df['low']
    df['close_open_diff'] = df['close'] - df['open']
    
    # Incorporate Volume Influence
    df['volume_adjusted_momentum'] = (df['high_low_range'] + df['close_open_diff']) * df['volume']
    
    # Adaptive Smoothing via Moving Average
    def dynamic_ema(data, span):
        return data.ewm(span=span, adjust=False).mean()
    
    # Calculate Daily Return
    df['daily_return'] = df['close'].pct_change()
    
    # Calculate Absolute Daily Return
    df['abs_daily_return'] = df['daily_return'].abs()
    
    # Calculate Robust Market Volatility
    df['robust_volatility'] = df['daily_return'].rolling(window=30).apply(lambda x: np.median(np.abs(x - np.median(x))))
    
    # Modify Volume-Adjusted Momentum with Robust Market Volatility
    df['vol_adj_mom_with_volatility'] = df['volume_adjusted_momentum'] / df['robust_volatility']
    
    # Incorporate Trend Reversal Signal
    df['short_term_mom'] = df['close'].ewm(span=5, adjust=False).mean()
    df['long_term_mom'] = df['close'].ewm(span=20, adjust=False).mean()
    df['mom_reversal'] = df['short_term_mom'] - df['long_term_mom']
    
    # Identify Reversal Points
    df['reversal_signal'] = np.sign(df['mom_reversal'])
    
    # Integrate Non-Linear Transformation
    df['sqrt_vol_adj_mom'] = np.sqrt(np.abs(df['vol_adj_mom_with_volatility']))
    df['log_vol_adj_mom'] = np.log1p(np.abs(df['vol_adj_mom_with_volatility']))
    
    # Enhance Reversal Signal with Adaptive Smoothing
    df['smoothed_reversal_signal'] = dynamic_ema(df['reversal_signal'], span=30)
    
    # Combine Smoothed Reversal Signal with Non-Linearly Transformed Momentum
    df['interim_alpha_factor'] = df['sqrt_vol_adj_mom'] + df['log_vol_adj_mom'] + df['smoothed_reversal_signal']
    
    # Final Adaptive Smoothing
    df['final_alpha_factor'] = dynamic_ema(df['interim_alpha_factor'], span=30)
    
    return df['final_alpha_factor'].dropna()

# Example usage:
# df = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)
# alpha_factor = heuristics_v2(df)
