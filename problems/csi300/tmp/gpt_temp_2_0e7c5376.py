import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic price and volume features
    data['range'] = data['high'] - data['low']
    data['gap'] = abs(data['open'] - data['close'].shift(1))
    data['close_open_diff'] = data['close'] - data['open']
    data['abs_close_open'] = abs(data['close_open_diff'])
    
    # Calculate peer averages (cross-sectional means for each day)
    peer_avg_range = data.groupby(data.index)['range'].transform('mean')
    peer_avg_gap = data.groupby(data.index)['gap'].transform('mean')
    
    # Volatility Structure
    data['relative_intraday_vol'] = data['range'] / peer_avg_range
    data['cross_asset_gap_eff'] = data['gap'] / peer_avg_gap
    
    # Volume calculations
    data['volume_5d_avg'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['relative_volume'] = data['volume'] / data['volume_5d_avg']
    peer_avg_rel_volume = data.groupby(data.index)['relative_volume'].transform('mean')
    data['relative_volume_divergence'] = data['relative_volume'] - peer_avg_rel_volume
    
    # Up/Down volume calculation
    data['up_volume'] = np.where(data['close'] > data['open'], data['volume'], 0)
    data['down_volume'] = np.where(data['close'] < data['open'], data['volume'], 0)
    
    daily_up_volume = data.groupby(data.index)['up_volume'].transform('sum')
    daily_down_volume = data.groupby(data.index)['down_volume'].transform('sum')
    data['sector_volume_pressure'] = ((daily_up_volume - daily_down_volume) / 
                                    (daily_up_volume + daily_down_volume + 1e-8)) * data['close_open_diff']
    
    # Microstructure Dynamics
    data['opening_range'] = (data['open'] - data['low']) / (data['range'] + 1e-8)
    peer_avg_opening_range = data.groupby(data.index)['opening_range'].transform('mean')
    data['opening_range_capture'] = data['opening_range'] * peer_avg_opening_range
    
    data['closing_efficiency'] = (data['abs_close_open'] / (data['range'] + 1e-8)) * np.sign(data['close_open_diff'])
    
    # Fractal Momentum calculations
    # Volume fractal (Hurst-like exponent approximation)
    def calculate_fractal(series, window=3):
        log_returns = np.log(series / series.shift(1)).dropna()
        if len(log_returns) < window:
            return np.nan
        rs = (log_returns.rolling(window).max() - log_returns.rolling(window).min()) / log_returns.rolling(window).std()
        return np.log(rs.mean()) / np.log(window) if not np.isnan(rs.mean()) else np.nan
    
    # Calculate volume and price fractals
    data['volume_fractal'] = data['volume'].rolling(window=5).apply(
        lambda x: calculate_fractal(pd.Series(x), 3), raw=False
    )
    
    data['price_fractal'] = data['close'].rolling(window=5).apply(
        lambda x: calculate_fractal(pd.Series(x), 3), raw=False
    )
    
    # Volume-Price Fractal correlation
    data['volume_price_fractal_corr'] = data['volume_fractal'].rolling(window=5).corr(data['price_fractal'])
    
    # Fractal Divergence
    peer_avg_volume_fractal = data.groupby(data.index)['volume_fractal'].transform('mean')
    data['fractal_divergence'] = data['volume_fractal'] - peer_avg_volume_fractal
    
    # Breakout Detection
    # Relative Breakout Momentum
    data['prev_close'] = data['close'].shift(1)
    data['breakout_momentum'] = np.where(
        data['high'] > data['prev_close'] * 1.01,
        (data['close'] - data['prev_close']) / data['prev_close'],
        0
    )
    peer_avg_breakout_momentum = data.groupby(data.index)['breakout_momentum'].transform('mean')
    data['breakout_strength'] = data['breakout_momentum'] / (peer_avg_breakout_momentum + 1e-8)
    
    # Volume Breakout Efficiency
    data['volume_breakout'] = np.where(
        data['volume'] > data['volume_5d_avg'] * 1.5,
        data['close_open_diff'] / (data['range'] + 1e-8),
        0
    )
    peer_avg_volume_breakout = data.groupby(data.index)['volume_breakout'].transform('mean')
    data['volume_breakout_efficiency'] = data['volume_breakout'] - peer_avg_volume_breakout
    
    # Signal Construction
    data['core_signal'] = (data['relative_volume_divergence'] * 
                          data['closing_efficiency'] * 
                          data['volume_price_fractal_corr'])
    
    data['breakout_multiplier'] = 1 + data['breakout_strength'] * 0.15
    data['volatility_multiplier'] = data['relative_intraday_vol'] * data['cross_asset_gap_eff']
    
    # Final Alpha
    data['raw_alpha'] = data['core_signal'] * data['breakout_multiplier'] * data['volatility_multiplier']
    data['final_alpha'] = data['raw_alpha'] * data['fractal_divergence']
    
    # Return the final alpha factor
    return data['final_alpha']
