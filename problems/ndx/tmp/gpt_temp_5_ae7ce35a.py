import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Compute Volume-Weighted High-Low Price Difference
    df['volume_weighted_high_low'] = (df['high'] - df['low']) * df['volume']
    
    # Calculate Deviation of Current Intraday High-Low Spread
    df['intraday_high_low_spread'] = df['high'] - df['low']
    intraday_high_low_mean = df['intraday_high_low_spread'].rolling(window=20).mean()
    df['deviation_intraday_high_low'] = df['intraday_high_low_spread'] - intraday_high_low_mean
    
    # Calculate Volume Weighted Average Price (VWAP)
    vwap_numerator = (df['open'] + df['high'] + df['low'] + df['close']) * df['volume']
    vwap_denominator = df['volume']
    df['vwap'] = vwap_numerator.cumsum() / vwap_denominator.cumsum()
    
    # Calculate Deviation of VWAP from Close
    df['deviation_vwap_close'] = df['vwap'] - df['close']
    
    # Incorporate Volume Impact Factor
    df['price_change'] = df['close'].diff().abs()
    df['volume_impact_factor'] = df['volume'] * df['price_change']
    
    # Integrate Historical High-Low Range and Momentum Contributions
    df['daily_volume_weighted_high_low_range'] = (df['high'] - df['low']) * df['volume']
    daily_volume_weighted_high_low_5_days = df['daily_volume_weighted_high_low_range'].rolling(window=5).sum()
    momentum_contributions = df['close'].pct_change(periods=10).rolling(window=5).sum()
    df['integrated_momentum_high_low'] = daily_volume_weighted_high_low_5_days * momentum_contributions
    
    # Adjust for Market Sentiment
    df['volatility_based_threshold'] = ((df['high'] - df['low']) / df['close']).rolling(window=5).mean()
    df['market_sentiment_adjustment'] = df['integrated_momentum_high_low'].apply(
        lambda x: x * 1.1 if x > df['volatility_based_threshold'] else x * 0.9
    )
    
    # Evaluate Overnight Sentiment
    df['overnight_return'] = (df['open'] / df['close'].shift(1)).apply(np.log)
    df['log_volume'] = df['volume'].apply(np.log)
    
    # Integrate Intraday and Overnight Signals
    df['average_intraday_return'] = (df['high'] / df['low'] + df['close'] / df['open']) / 2
    df['intraday_overnight_diff'] = df['average_intraday_return'] - df['overnight_return']
    df['volume_adjusted_indicator'] = df['volume'] - df['volume'].rolling(window=20).mean()
    df['integrated_intraday_overnight_signal'] = df['intraday_overnight_diff'] * df['volume_adjusted_indicator']
    
    # Generate Alpha Factor
    df['alpha_factor'] = df['intraday_high_low_spread'] / df['volume']
    
    # Combine Indicators
    df['combined_indicators'] = df['deviation_intraday_high_low'] + df['deviation_vwap_close'] + df['alpha_factor'] * df['close']
    
    # Synthesize Overall Alpha Factor
    df['overall_alpha_factor'] = (
        df['integrated_momentum_high_low'] + 
        df['integrated_intraday_overnight_signal'] + 
        df['combined_indicators'] * df['volume_adjusted_indicator']
    )
    
    return df['overall_alpha_factor']

# Example usage:
# df = pd.read_csv('your_data.csv', parse_dates=True, index_col=0)
# alpha_factor = heuristics_v2(df)
