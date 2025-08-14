import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Volume-Weighted Intraday Return
    df['volume_weighted_intraday_return'] = (df['high'] - df['low']) * df['volume']
    
    # Calculate Intraday Momentum
    df['intraday_momentum'] = (df['high'] - df['low']) / df['open']
    
    # Measure Volume Stability
    df['volume_change'] = df['volume'].diff().abs()
    df['volume_stability'] = df['volume_change'].rolling(window=5).sum()
    
    # Calculate Volume Flow
    df['volume_flow'] = (df['close'] * df['volume'] - df['open'] * df['volume']) / ((df['open'] + df['close']) / 2)
    
    # Combine Intraday Momentum and Volume-Weighted Intraday Return
    df['combined_momentum'] = df['intraday_momentum'] * df['volume_weighted_intraday_return']
    
    # Integrate Price Trend
    df['price_trend'] = df['close'].diff()
    
    # Incorporate Trading Volume and Amount
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['amount_weighted_price'] = (df['close'] * df['amount']).cumsum() / df['amount'].cumsum()
    
    # Adjusted Reversal Indicator
    df['lagged_volume_weighted_intraday_return'] = df['volume_weighted_intraday_return'].shift(1)
    df['reversal_indicator'] = df['volume_weighted_intraday_return'] - df['lagged_volume_weighted_intraday_return']
    
    # Combine VWAP and Intraday Momentum
    df['vwap_momentum'] = (df['vwap'] - df['open']) / df['open']
    df['avg_momentum'] = (df['vwap_momentum'] + df['intraday_momentum']) / 2
    
    # Weight by Intraday Volatility (ATR over 5 days)
    df['true_range'] = df[['high' - 'low', (df['high'] - df['close']).abs(), (df['low'] - df['close']).abs()]].max(axis=1)
    df['atr_5'] = df['true_range'].rolling(window=5).mean()
    df['weighted_momentum'] = df['combined_momentum'] * df['atr_5']
    
    # Final Alpha Factor Combination
    df['final_alpha_factor'] = (df['combined_momentum'] * df['vwap']) * df['price_trend']
    
    # Summarize Momentum Over Multiple Days
    df['cumulative_momentum'] = df['final_alpha_factor'].rolling(window=10).sum()
    
    return df['cumulative_momentum']
