import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Ensure data is sorted by date
    data = data.sort_index()
    
    # Calculate basic components
    data['prev_close'] = data['close'].shift(1)
    data['daily_return'] = data['close'] - data['prev_close']
    data['daily_range'] = data['high'] - data['low']
    data['intraday_move'] = data['close'] - data['open']
    
    # Volatility Structure
    # Intraday Volatility Ratio
    data['intraday_vol_ratio'] = data['daily_range'] / np.abs(data['close'] - data['prev_close']).replace(0, np.nan)
    
    # Volatility Persistence
    data['vol_5d'] = data['daily_return'].rolling(window=5, min_periods=3).std()
    data['vol_10d'] = data['daily_return'].rolling(window=10, min_periods=5).std()
    data['return_5d'] = data['close'].pct_change(5)
    data['volatility_persistence'] = (data['vol_5d'] / data['vol_10d']) * np.sign(data['return_5d'])
    
    # Volume-Price Integration
    # Volume-Volatility Divergence
    data['avg_volume_5d'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['avg_range_5d'] = data['daily_range'].rolling(window=5, min_periods=3).mean()
    data['volume_volatility_divergence'] = (data['volume'] / data['avg_volume_5d']) - (data['daily_range'] / data['avg_range_5d'])
    
    # Directional Volume Pressure
    data['up_volume'] = np.where(data['close'] > data['open'], data['volume'], 0)
    data['down_volume'] = np.where(data['close'] < data['open'], data['volume'], 0)
    data['directional_volume_pressure'] = ((data['up_volume'] - data['down_volume']) / 
                                         (data['up_volume'] + data['down_volume']).replace(0, np.nan)) * data['intraday_move']
    
    # Microstructure Dynamics
    # Opening Range Capture
    data['opening_range_capture'] = ((data['open'] - data['low']) / 
                                   data['daily_range'].replace(0, np.nan)) * data['intraday_move']
    
    # Closing Range Efficiency
    data['closing_range_efficiency'] = (np.abs(data['intraday_move']) / 
                                      data['daily_range'].replace(0, np.nan)) * np.sign(data['intraday_move'])
    
    # Regime-Adaptive Momentum
    data['vol_20d'] = data['daily_return'].rolling(window=20, min_periods=10).std()
    data['high_vol_momentum'] = data['return_5d'] * (data['vol_5d'] / data['vol_20d'])
    data['low_vol_momentum'] = data['return_5d'] * (data['vol_20d'] / data['vol_5d'])
    
    # Fractal Integration
    # Volume Fractal (Hurst-like calculation for volume)
    def hurst_like_volume(series):
        if len(series) < 3:
            return np.nan
        log_vol = np.log(series)
        return (log_vol.iloc[-1] - log_vol.iloc[0]) / (np.std(log_vol) * np.sqrt(len(series)))
    
    # Price Fractal (Hurst-like calculation for price)
    def hurst_like_price(series):
        if len(series) < 3:
            return np.nan
        log_price = np.log(series)
        return (log_price.iloc[-1] - log_price.iloc[0]) / (np.std(log_price) * np.sqrt(len(series)))
    
    # Calculate 3-day fractals
    data['volume_fractal_3d'] = data['volume'].rolling(window=3, min_periods=3).apply(
        hurst_like_volume, raw=False)
    data['price_fractal_3d'] = data['close'].rolling(window=3, min_periods=3).apply(
        hurst_like_price, raw=False)
    
    # Volume-Price Fractal Correlation (5-day rolling correlation)
    data['volume_price_fractal_corr'] = data['volume_fractal_3d'].rolling(
        window=5, min_periods=3).corr(data['price_fractal_3d'])
    
    # Volume-Position Alignment
    data['min_low_3d'] = data['low'].rolling(window=3, min_periods=3).min()
    data['max_high_3d'] = data['high'].rolling(window=3, min_periods=3).max()
    data['volume_position_alignment'] = ((data['close'] - data['min_low_3d']) / 
                                       (data['max_high_3d'] - data['min_low_3d']).replace(0, np.nan)) * data['volume_fractal_3d']
    
    # Adaptive Alpha Generation
    # Core Signal
    data['core_signal'] = (data['volatility_persistence'] * 
                          data['volume_volatility_divergence'] * 
                          data['closing_range_efficiency'])
    
    # Regime Multiplier
    high_vol_condition = (data['intraday_vol_ratio'] > 2) & (data['volume'] > 1.5 * data['avg_volume_5d'])
    low_vol_condition = (data['intraday_vol_ratio'] < 0.5) & (data['volume'] < 0.8 * data['avg_volume_5d'])
    
    data['regime_multiplier'] = np.where(high_vol_condition, 1.3, 
                                       np.where(low_vol_condition, 0.7, 1.0))
    
    # Final Alpha
    data['final_alpha'] = (data['core_signal'] * 
                          data['regime_multiplier'] * 
                          data['volume_price_fractal_corr'])
    
    # Return the final alpha factor series
    return data['final_alpha']
