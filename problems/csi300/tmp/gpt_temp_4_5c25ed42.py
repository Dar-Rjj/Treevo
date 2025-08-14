import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Return
    df['intraday_return'] = df['close'] - df['open']
    
    # Calculate High-Low Price Range
    df['high_low_range'] = df['high'] - df['low']
    
    # Calculate Volume Adjusted Intraday Return
    df['vol_adj_intraday_return'] = df['intraday_return'] * df['volume']
    
    # Calculate Volume Adjusted High-Low Range
    df['vol_adj_high_low_range'] = df['high_low_range'] * df['volume']
    
    # Calculate Smoothed Intraday Momentum
    df['smoothed_intraday_momentum'] = df['vol_adj_intraday_return'].rolling(window=10).mean()
    recent_volume = df['volume'].rolling(window=5).mean()
    df['weighted_smoothed_intraday_momentum'] = df['smoothed_intraday_momentum'] * (recent_volume / df['volume'])
    
    # Calculate True Range
    df['true_range'] = df[['high_low_range', 'high_close_prev', 'low_close_prev']].max(axis=1)
    
    # Calculate Volume Weighted Average True Range (VWATR)
    df['vwatr'] = df['true_range'] * df['volume']
    
    # Calculate High-Low Breakout Potential
    df['breakout_potential'] = df['vol_adj_high_low_range'].rolling(window=20).mean()
    recent_volume_long = df['volume'].rolling(window=10).mean()
    df['weighted_breakout_potential'] = df['breakout_potential'] * (recent_volume_long / df['volume'])
    
    # Calculate Price Momentum
    df['price_momentum'] = df['close'] - df['close'].shift(10)
    
    # Adjust Price Momentum by VWATR
    df['adj_price_momentum'] = df['price_momentum'] / df['vwatr']
    
    # Calculate Momentum Reversal Indicator
    df['momentum_reversal_indicator'] = (df['intraday_return'] - df['weighted_smoothed_intraday_momentum']) / df['weighted_breakout_potential']
    
    # Calculate Volume Weighted Average Price (VWAP)
    df['typical_price'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    df['vwap'] = (df['volume'] * df['typical_price']).cumsum() / df['volume'].cumsum()
    
    # Combine Intraday Range and VWAP
    df['high_vwap_diff'] = df['high'] - df['vwap']
    df['vwap_low_diff'] = df['vwap'] - df['low']
    df['vwap_based_intraday_range'] = df['high_vwap_diff'] + df['vwap_low_diff']
    
    # Final Alpha Factor
    df['final_alpha_factor'] = (df['adj_price_momentum'] * df['weighted_breakout_potential'] 
                                + df['momentum_reversal_indicator'] 
                                + df['vwap_based_intraday_range'])

    return df['final_alpha_factor']
