import pandas as pd
import pandas as pd

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
    recent_vol_avg = df['volume'].rolling(window=5).mean()
    df['smoothed_intraday_momentum'] *= recent_vol_avg
    
    # Calculate Volume Weighted Average True Range (VWATR)
    df['true_range'] = df[['high' - 'low', 'high' - df['close'].shift(1), df['close'].shift(1) - 'low']].max(axis=1)
    df['vwatr'] = df['true_range'] * df['volume']
    
    # Calculate High-Low Breakout Potential
    df['hl_breakout_potential'] = df['vol_adj_high_low_range'].rolling(window=20).mean()
    recent_vol_avg_10 = df['volume'].rolling(window=10).mean()
    df['hl_breakout_potential'] *= recent_vol_avg_10
    
    # Calculate Price Momentum
    df['price_momentum'] = df['close'] - df['close'].shift(10)
    
    # Adjust Price Momentum by VWATR
    df['adj_price_momentum'] = df['price_momentum'] / df['vwatr']
    
    # Calculate Momentum Reversal Indicator
    df['momentum_reversal_indicator'] = (df['intraday_return'] - df['smoothed_intraday_momentum']) / df['hl_breakout_potential']
    
    # Calculate Volume Weighted Average Price (VWAP)
    df['typical_price'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    df['vwap'] = (df['typical_price'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Combine Intraday Range and VWAP
    df['vwap_high_diff'] = df['high'] - df['vwap']
    df['vwap_low_diff'] = df['vwap'] - df['low']
    df['combined_vwap_range'] = df['vwap_high_diff'] + df['vwap_low_diff']
    
    # Apply a moving average to smooth the factor
    df['smoothed_factor'] = df['combined_vwap_range'].rolling(window=10).mean()
    
    # Calculate Volume Adjusted Momentum
    df['volume_ratio'] = df['volume'] / df['volume'].shift(10)
    df['vol_adj_momentum'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    df['vol_adj_momentum'] *= df['volume_ratio']
    
    # Combine Momentum and Volume Ratio
    df['combined_momentum'] = df['price_momentum'] + df['vol_adj_momentum']
    
    # Final Alpha Factor
    df['alpha_factor'] = (df['adj_price_momentum'] * df['hl_breakout_potential']) + df['momentum_reversal_indicator'] + df['combined_momentum'] + df['smoothed_factor']
    
    # Enhanced Volume Adjusted Momentum
    df['enhanced_momentum'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    df['enhanced_volume_ratio'] = df['volume'] / df['volume'].shift(10)
    df['enhanced_vol_adj_momentum'] = df['enhanced_momentum'] * df['enhanced_volume_ratio']
    
    # Add Enhanced Volume Adjusted Momentum to the final alpha factor
    df['final_alpha_factor'] = df['alpha_factor'] + df['enhanced_vol_adj_momentum']
    
    return df['final_alpha_factor'].dropna()
