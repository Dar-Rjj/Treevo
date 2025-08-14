import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Volume-Weighted High-Low Price Difference
    df['vol_weighted_high_low'] = (df['high'] - df['low']) * df['volume']
    
    # Daily Price Change
    df['daily_price_change'] = df['close'].diff()
    
    # Volume Impact Factor
    df['volume_impact_factor'] = df['volume'] * df['daily_price_change'].abs()
    
    # Historical High-Low Range and Momentum
    df['vol_weighted_high_low_5d_sum'] = df['vol_weighted_high_low'].rolling(window=5).sum()
    df['momentum_5d'] = df['daily_price_change'].rolling(window=5).mean()
    df['historical_high_low_momentum'] = df['vol_weighted_high_low_5d_sum'] * df['momentum_5d']
    
    # Market Sentiment Adjustment
    df['volatility_based_threshold'] = ((df['high'] - df['low']) / df['close']).rolling(window=5).mean()
    df['market_sentiment_adjustment'] = df.apply(
        lambda row: 1.2 if row['historical_high_low_momentum'] > row['volatility_based_threshold'] else 0.8, axis=1
    )
    
    # Overnight Sentiment
    df['overnight_return'] = df['open'] / df['close'].shift(1)
    df['log_volume'] = df['volume'].apply(lambda x: pd.np.log(x))
    df['overnight_sentiment'] = df['log_volume'] * df['overnight_return']
    
    # Intraday and Overnight Signals
    df['intraday_signal'] = (df[['high', 'low']].mean(axis=1) / df['close']).sub(df['overnight_return'])
    df['recent_volume_vs_ma'] = df['volume'] - df['volume'].rolling(window=5).mean()
    
    # Volume Trend and Reversal Potential
    df['volume_direction'] = df.apply(
        lambda row: 1 if row['volume'] > row['volume'].shift(1) else -1, axis=1
    )
    df['intraday_high_low_diff'] = df['high'] - df['low']
    df['weighted_reversal_potential'] = df['intraday_high_low_diff'] * df['volume_direction']
    df['reversal_adjustment'] = df['weighted_reversal_potential'] * df['daily_price_change'].diff()
    
    # Integrated Alpha Factor
    df['integrated_alpha_factor'] = (
        df['vol_weighted_high_low'] +
        df['volume_impact_factor'] +
        df['historical_high_low_momentum'] * df['market_sentiment_adjustment'] +
        df['overnight_sentiment'] +
        df['intraday_signal'] * df['recent_volume_vs_ma'] +
        df['reversal_adjustment']
    )
    
    return df['integrated_alpha_factor']

# Example usage:
# df = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
