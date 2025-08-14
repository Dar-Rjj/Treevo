import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Compute Short-Term Momentum
    df['10_day_ma_close'] = df['close'].rolling(window=10).mean()
    df['short_term_momentum'] = df['close'] - df['10_day_ma_close']
    
    # Compute Long-Term Momentum
    df['20_day_ma_close'] = df['close'].rolling(window=20).mean()
    df['long_term_momentum'] = df['close'] - df['20_day_ma_close']
    
    # Determine Relative Strength Score
    df['relative_strength_score'] = df['short_term_momentum'] / (df['long_term_momentum'] + 1e-8)
    
    # Calculate Intraday Volatility
    df['intraday_volatility'] = (df['high'] - df['low']) / df['close']
    
    # Combine Momentum and Intraday Volatility
    df['momentum_volatility'] = df['relative_strength_score'] * df['intraday_volatility']
    
    # Calculate Volume Trend Impact
    df['10_day_ma_volume'] = df['volume'].rolling(window=10).mean()
    df['20_day_ma_volume'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['10_day_ma_volume'] / (df['20_day_ma_volume'] + 1e-8)
    df['volume_ratio_score'] = np.where(df['volume_ratio'] > 1, 1, 0)
    
    # Dynamic Weighting Adjustment
    df['combined_scores'] = df['relative_strength_score'] * df['volume_ratio_score']
    df['final_dynamic_score'] = 1 - df['combined_scores']
    
    # Incorporate Intraday Factors
    df['intraday_open_close_return'] = (df['close'] - df['open']) / df['open']
    df['intraday_high_low_spread'] = (df['high'] - df['low']) / df['low']
    df['volume_adjusted_intraday_open_close'] = df['intraday_open_close_return'] * df['volume']
    df['volume_adjusted_intraday_high_low'] = df['intraday_high_low_spread'] * df['volume']
    df['volume_adjusted_intraday_factors'] = df['volume_adjusted_intraday_open_close'] + df['volume_adjusted_intraday_high_low']
    
    # Integrate Adjusted Momentum and Intraday Dynamics
    df['integrated_momentum_volatility_intraday'] = df['momentum_volatility'] * df['volume_adjusted_intraday_factors']
    
    # Smooth Over N Days
    df['smoothed_integrated_factor'] = df['integrated_momentum_volatility_intraday'].rolling(window=10).mean()
    
    # Compute Price Momentum
    df['10_day_return'] = df['close'].pct_change(periods=10)
    
    # Adjust Price Momentum by Intraday Volatility
    df['adjusted_momentum'] = df['10_day_return'] / (df['intraday_volatility'] + 1e-8)
    
    # Compute Open Price Trend
    open_prices = df['open'].values
    dates = df.index
    slopes = [linregress(dates[i-10:i+1], open_prices[i-10:i+1])[0] for i in range(10, len(dates))]
    df['open_price_trend_slope'] = pd.Series(slopes, index=dates[10:])
    
    # Intraday Range Momentum
    df['intraday_range_current'] = (df['high'] - df['low']) / df['low']
    df['intraday_range_previous'] = df['intraday_range_current'].shift(1)
    df['intraday_range_momentum'] = df['intraday_range_current'] - df['intraday_range_previous']
    
    # Volume Confirmation
    df['volume_change'] = df['volume'] - df['volume'].shift(1)
    
    # Final Alpha Factor
    df['alpha_factor'] = (
        df['integrated_momentum_volatility_intraday'] * df['final_dynamic_score'] +
        df['adjusted_momentum'] + 
        df['intraday_range_momentum'] + 
        df['volume_adjusted_intraday_factors']
    ) * df['volume_change'] * df['open_price_trend_slope']
    
    return df['alpha_factor']
