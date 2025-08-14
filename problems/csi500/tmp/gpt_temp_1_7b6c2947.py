import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, gdp_growth, inflation_rate, fed_funds_rate, treasury_yield, sentiment_scores):
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = df['open'].shift(-1) / df['close'] - 1
    
    # Weight by Volume
    df['volume_weighted_return'] = df['close_to_open_return'] * df['volume']
    
    # Dynamic Lookback Periods with Adaptive Adjustments
    short_term_lookback = 5
    mid_term_lookback = 10
    long_term_lookback = 20
    
    # Calculate Simple Moving Averages and Standard Deviations
    df['short_term_sma'] = df['volume_weighted_return'].rolling(window=short_term_lookback).mean()
    df['short_term_std'] = df['volume_weighted_return'].rolling(window=short_term_lookback).std()
    
    df['mid_term_sma'] = df['volume_weighted_return'].rolling(window=mid_term_lookback).mean()
    df['mid_term_std'] = df['volume_weighted_return'].rolling(window=mid_term_lookback).std()
    
    df['long_term_sma'] = df['volume_weighted_return'].rolling(window=long_term_lookback).mean()
    df['long_term_std'] = df['volume_weighted_return'].rolling(window=long_term_lookback).std()
    
    # Volatility Regime Switching
    def volatility_regime(std):
        if std > std.quantile(0.75):
            return 'High'
        elif std < std.quantile(0.25):
            return 'Low'
        else:
            return 'Medium'
    
    df['volatility_regime'] = df['short_term_std'].apply(volatility_regime)
    
    # Adjust Lookback Periods Based on Volatility Regime
    df['adjusted_lookback'] = df.apply(lambda row: short_term_lookback if row['volatility_regime'] == 'High' else 
                                                    (mid_term_lookback if row['volatility_regime'] == 'Medium' else 
                                                     long_term_lookback), axis=1)
    
    # Adjust Lookback Periods Based on Sentiment
    df['sentiment_adjusted_lookback'] = df.apply(lambda row: row['adjusted_lookback'] - 1 if sentiment_scores[row.name] > 0 else 
                                                              (row['adjusted_lookback'] + 1 if sentiment_scores[row.name] < 0 else 
                                                               row['adjusted_lookback']), axis=1)
    
    # Combine Multi-Period Volatility
    df['combined_volatility'] = (df['short_term_std'] + df['mid_term_std'] + df['long_term_std']) / 3
    
    # Integrate Market Signals
    df['high_low_range'] = df['high'] - df['low']
    df['volume_change'] = df['volume'] - df['volume'].shift(1)
    
    # Incorporate Macroeconomic Indicators
    df['gdp_growth'] = gdp_growth
    df['inflation_rate'] = inflation_rate
    df['fed_funds_rate'] = fed_funds_rate
    df['treasury_yield'] = treasury_yield
    
    # Final Factor
    df['final_factor'] = df['volume_weighted_return'].rolling(window=df['sentiment_adjusted_lookback']).mean() / df['combined_volatility']
    df['nonlinear_transformed_factor'] = np.log(np.sqrt(df['final_factor']))
    
    # Output Final Alpha Factor
    return df['nonlinear_transformed_factor']

# Example usage:
# df = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)
# gdp_growth, inflation_rate, fed_funds_rate, treasury_yield = 0.02, 0.03, 0.04, 0.05
# sentiment_scores = pd.read_csv('sentiment_scores.csv', index_col='date', parse_dates=True)
# alpha_factor = heuristics_v2(df, gdp_growth, inflation_rate, fed_funds_rate, treasury_yield, sentiment_scores)
