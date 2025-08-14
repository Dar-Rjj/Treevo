import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, short_window=20, long_window=252, vol_window=20, pct_change_window=10, macro_window=90):
    # Calculate Simple Moving Average (SMA) of Close Prices
    sma_short = df['close'].rolling(window=short_window).mean()
    
    # Compute Volume-Adjusted Volatility
    high_low_diff = df['high'] - df['low']
    vol_adj_volatility = (high_low_diff * df['volume']).rolling(window=vol_window).mean()
    
    # Compute Price Momentum
    price_momentum = (df['close'] - sma_short) / df['close'].rolling(window=short_window).mean()
    
    # Incorporate Additional Price Change Metrics
    close_pct_change = df['close'].pct_change(pct_change_window)
    high_low_range = df['high'] - df['low']
    
    # Consider Market Trend Alignment
    sma_long = df['close'].rolling(window=long_window).mean()
    trend_indicator = np.where(sma_short > sma_long, 1, -1)
    
    # Integrate Macroeconomic Indicators
    # Assuming 'macro_data' is a DataFrame with the same index and columns for GDP Growth and Inflation Rate
    # Example: macro_data = pd.DataFrame({'GDP Growth': [...], 'Inflation Rate': [...]}, index=df.index)
    gdp_growth = macro_data['GDP Growth']
    inflation_rate = macro_data['Inflation Rate']
    macro_impact = (gdp_growth.rolling(window=macro_window).corr(df['close']) + 
                    inflation_rate.rolling(window=macro_window).corr(df['close'])) / 2
    
    # Final Alpha Factor
    weights = {
        'price_momentum': 0.4,
        'vol_adj_volatility': -0.3,
        'close_pct_change': 0.2,
        'high_low_range': 0.1
    }
    
    # Adjust Weights Based on Market Trend
    weights['price_momentum'] *= trend_indicator
    weights['vol_adj_volatility'] *= trend_indicator
    weights['close_pct_change'] *= trend_indicator
    weights['high_low_range'] *= trend_indicator
    
    # Adjust Weights Based on Macroeconomic Impact
    macro_weights = np.where(macro_impact > 0, 1.5, 0.5)
    alpha_factor = (weights['price_momentum'] * price_momentum +
                    weights['vol_adj_volatility'] * vol_adj_volatility +
                    weights['close_pct_change'] * close_pct_change +
                    weights['high_low_range'] * high_low_range) * macro_weights
    
    return alpha_factor

# Example usage:
# df = pd.read_csv('market_data.csv', parse_dates=True, index_col='date')
# macro_data = pd.read_csv('macroeconomic_data.csv', parse_dates=True, index_col='date')
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
