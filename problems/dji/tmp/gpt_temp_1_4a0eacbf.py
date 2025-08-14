import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Compute Short-Term Momentum
    df['short_momentum'] = df['close'] - df['close'].rolling(window=10).mean()
    
    # Compute Long-Term Momentum
    df['long_momentum'] = df['close'] - df['close'].rolling(window=20).mean()
    
    # Determine Relative Strength Score
    df['relative_strength_score'] = df['short_momentum'] / df['long_momentum']
    
    # Calculate Intraday Volatility
    df['intraday_volatility'] = (df['high'] - df['low']) / df['close']
    
    # Combine Momentum and Intraday Volatility
    df['momentum_volatility'] = df['relative_strength_score'] * df['intraday_volatility']
    
    # Calculate Volume Trend Impact
    df['short_volume_trend'] = df['volume'].rolling(window=10).mean()
    df['long_volume_trend'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['short_volume_trend'] / df['long_volume_trend']
    
    # Assign Volume Ratio Score
    df['volume_ratio_score'] = df['volume_ratio'].apply(lambda x: 1 if x > 1 else 0)
    
    # Dynamic Weighting Adjustment
    df['combined_scores'] = df['relative_strength_score'] * df['volume_ratio_score']
    df['final_dynamic_score'] = 1 - df['combined_scores']
    
    # Incorporate Intraday Factors
    df['intraday_open_close_return'] = (df['close'] - df['open']) / df['open']
    df['intraday_high_low_spread'] = (df['high'] - df['low'])
    df['volume_adjusted_intraday_open_close'] = df['intraday_open_close_return'] * df['volume']
    df['volume_adjusted_intraday_high_low'] = df['intraday_high_low_spread'] * df['volume']
    df['volume_adjusted_intraday_factors'] = df['volume_adjusted_intraday_open_close'] + df['volume_adjusted_intraday_high_low']
    
    # Integrate Adjusted Momentum and Intraday Dynamics
    df['integrated_momentum_intraday'] = df['momentum_volatility'] * df['volume_adjusted_intraday_factors']
    
    # Smooth Over N Days
    df['smoothed_factor'] = df['integrated_momentum_intraday'].rolling(window=10).mean()
    
    # Compute Price Momentum
    df['price_momentum'] = df['close'].pct_change(periods=10)
    
    # Adjust Price Momentum by Intraday Volatility
    df['adjusted_momentum'] = df['price_momentum'] / df['intraday_volatility']
    
    # Compute Open Price Trend
    open_prices = df['open'].values
    dates = df.index.values
    slopes = [linregress(dates[i-9:i+1], open_prices[i-9:i+1]).slope for i in range(9, len(dates))]
    df['open_price_trend_slope'] = np.nan
    df.loc[df.index[9:], 'open_price_trend_slope'] = slopes
    
    # Intraday Range Momentum
    df['current_intraday_range'] = df['high'] - df['low']
    df['previous_intraday_range'] = df['current_intraday_range'].shift(1)
    df['intraday_range_momentum'] = df['current_intraday_range'] - df['previous_intraday_range']
    
    # Volume Confirmation
    df['volume_change'] = df['volume'] - df['volume'].shift(1)
    
    # Enhanced Volume Impact
    df['volume_momentum'] = df['volume'] - df['volume'].rolling(window=10).mean()
    
    # Final Alpha Factor
    df['alpha_factor'] = (df['smoothed_factor'] * df['final_dynamic_score'] +
                          df['adjusted_momentum'] + df['intraday_range_momentum'] +
                          df['volume_adjusted_intraday_factors']) * df['volume_change'] * df['open_price_trend_slope']
    
    return df['alpha_factor']
