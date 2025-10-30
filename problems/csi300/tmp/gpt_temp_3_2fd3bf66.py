import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate rolling statistics
    data['volume_5d_avg'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_10d_avg'] = data['volume'].rolling(window=10, min_periods=5).mean()
    data['volume_20d_avg'] = data['volume'].rolling(window=20, min_periods=10).mean()
    data['volume_20d_std'] = data['volume'].rolling(window=20, min_periods=10).std()
    
    # Calculate ATR (Average True Range)
    data['tr'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['atr_5d'] = data['tr'].rolling(window=5, min_periods=3).mean()
    data['atr_10d'] = data['tr'].rolling(window=10, min_periods=5).mean()
    data['atr_20d'] = data['tr'].rolling(window=20, min_periods=10).mean()
    
    # Calculate price ranges
    data['high_5d'] = data['high'].rolling(window=5, min_periods=3).max()
    data['low_5d'] = data['low'].rolling(window=5, min_periods=3).min()
    data['high_10d'] = data['high'].rolling(window=10, min_periods=5).max()
    data['low_10d'] = data['low'].rolling(window=10, min_periods=5).min()
    data['high_20d'] = data['high'].rolling(window=20, min_periods=10).max()
    data['low_20d'] = data['low'].rolling(window=20, min_periods=10).min()
    
    # Calculate volume/price ratio statistics
    data['vol_price_ratio'] = data['volume'] / data['close']
    data['vol_price_10d_median'] = data['vol_price_ratio'].rolling(window=10, min_periods=5).median()
    
    # Calculate volume slope
    def volume_slope(series):
        if len(series) < 2:
            return np.nan
        x = np.arange(len(series))
        return np.polyfit(x, series, 1)[0]
    
    data['volume_slope_3d'] = data['volume'].rolling(window=3, min_periods=2).apply(volume_slope, raw=True)
    
    # Calculate individual components
    # Momentum & Volume Acceleration
    mom_vol_acc = (data['close'] / data['close'].shift(5) - 1) * data['volume_slope_3d']
    
    # Intraday Persistence
    intraday_persistence = (abs((data['high'] - data['low']) / (data['open'] - data['close']).replace(0, np.nan)) * 
                           (data['volume'] / data['volume_5d_avg']) * 
                           np.sign(data['close'] - (data['high'] + data['low']) / 2))
    
    # Liquidity-Adjusted Reversal
    liq_adj_reversal = ((data['close'] / data['close'].shift(3) - 1) * 
                       (data['volume'] / data['close']) / 
                       data['vol_price_10d_median'])
    
    # Volatility Breakout
    vol_breakout = ((data['atr_10d'] / data['atr_20d']) * 
                   ((data['volume'] - data['volume_20d_avg']) / data['volume_20d_std']) * 
                   np.sign(data['close'] - data['open']))
    
    # Price Efficiency
    price_efficiency = ((data['high'] - data['low']) / 
                       abs(data['close'] - data['open']).replace(0, np.nan) / 
                       ((data['volume'] * data['close']) / 
                       (data['volume'] * data['close']).rolling(window=10, min_periods=5).mean()))
    
    # Order Flow Proxy
    def order_flow_calc(window):
        if len(window) < 3:
            return np.nan
        close_low_ratio = (window['close'] - window['low']) / (window['high'] - window['low']).replace(0, np.nan)
        vol_ratio = window['volume'] / data['volume_10d_avg'].loc[window.index]
        return (close_low_ratio * vol_ratio).sum()
    
    order_flow = pd.Series(index=data.index, dtype=float)
    for i in range(3, len(data)):
        if i >= 3:
            window_data = data.iloc[i-3:i+1]
            order_flow.iloc[i] = order_flow_calc(window_data)
    
    # Volume Anomaly
    volume_anomaly = ((data['volume'] / data['volume_20d_avg']) * 
                     (data['close'] / data['close'].shift(1) - 1))
    
    # Multi-Timeframe Momentum
    multi_timeframe_mom = (((data['close'] / data['close'].shift(3) - 1) / 
                          (data['high_10d'] - data['low_10d'])) * 
                         ((data['close'] / data['close'].shift(10) - 1) / 
                          (data['high_20d'] - data['low_20d'])) * 
                         np.sign((data['close'] / data['close'].shift(3) - 1) * 
                                (data['close'] / data['close'].shift(10) - 1)))
    
    # Gap Momentum Divergence
    gap_mom_div = ((data['open'] / data['close'].shift(1) - 1) * 
                  (data['close'] / data['open'] - 1) * 
                  data['volume'] / data['volume_5d_avg'])
    
    # Volatility Compression Breakout
    vol_comp_breakout = ((data['atr_5d'] / data['atr_10d']) * 
                        (data['close'] / data['close'].shift(1) - 1) * 
                        data['volume'] / data['volume_10d_avg'])
    
    # Relative Strength Efficiency
    rel_strength_eff = (((data['close'] / data['close'].shift(5) - 1) / 
                        (data['high_5d'] - data['low_5d'])) * 
                       (data['volume'] / data['volume_5d_avg']))
    
    # Price-Volume Consistency
    price_vol_consistency = (np.sign(data['close'] - data['close'].shift(1)) * 
                           np.sign(data['volume'] - data['volume_5d_avg']) * 
                           abs(data['close'] / data['close'].shift(1) - 1) * 
                           data['volume'] / data['volume_5d_avg'])
    
    # Combine all factors with equal weights
    factors = [
        mom_vol_acc, intraday_persistence, liq_adj_reversal, vol_breakout,
        price_efficiency, order_flow, volume_anomaly, multi_timeframe_mom,
        gap_mom_div, vol_comp_breakout, rel_strength_eff, price_vol_consistency
    ]
    
    # Normalize and combine
    valid_factors = []
    for f in factors:
        if f.notna().any():
            # Remove outliers and normalize
            f_clean = f.clip(lower=f.quantile(0.01), upper=f.quantile(0.99))
            f_norm = (f_clean - f_clean.mean()) / f_clean.std()
            valid_factors.append(f_norm)
    
    # Equal-weighted combination
    if valid_factors:
        combined_factor = sum(valid_factors) / len(valid_factors)
    else:
        combined_factor = pd.Series(index=data.index, dtype=float)
    
    return combined_factor
