import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Compute Short-Term and Long-Term Momentum
    short_term_mom = df['close'] - df['close'].rolling(window=10).mean()
    long_term_mom = df['close'] - df['close'].rolling(window=20).mean()
    
    # Compute Short-Term and Long-Term Volume Trend
    short_term_vol_trend = df['volume'].rolling(window=10).mean()
    long_term_vol_trend = df['volume'].rolling(window=20).mean()
    
    # Determine Relative Strength Score
    relative_strength_score = (short_term_mom > long_term_mom).astype(int)
    
    # Determine Volume Ratio Score
    volume_ratio = short_term_vol_trend / long_term_vol_trend
    volume_ratio_score = (volume_ratio > 1).astype(int)
    
    # Combine Scores
    final_dynamic_score = 1 - (relative_strength_score * volume_ratio_score)
    
    # Apply Dynamic Score to Close Price
    alpha_factor_dynamic = df['close'] * final_dynamic_score
    
    # Calculate Intraday High-Low Spread
    intraday_high_low_spread = df['high'] - df['low']
    
    # Calculate Intraday Open-Close Return
    intraday_open_close_return = df['close'] - df['open']
    
    # Calculate Combined Intraday Factors
    combined_intraday_factors = (intraday_high_low_spread - intraday_open_close_return) * df['volume']
    
    # Calculate Price Momentum
    daily_returns = df['close'].pct_change()
    price_momentum = daily_returns.rolling(window=10).sum()
    smoothed_price_momentum = price_momemtum.ewm(span=10, adjust=False).mean()
    
    # Adjust by Volume and Price Momentum
    adjusted_intraday_factors = combined_intraday_factors * df['volume']
    intermediate_alpha_factor = adjusted_intraday_factors + smoothed_price_momentum
    
    # Integrate Volume Impact
    avg_volume = df['volume'].rolling(window=10).mean()
    volume_trend = df['volume'].pct_change(10)
    volume_adjusted_alpha_factor = intermediate_alpha_factor * (1 + volume_trend)
    
    # Final Alpha Factor
    final_alpha_factor = volume_adjusted_alpha_factor * alpha_factor_dynamic
    
    return final_alpha_factor
