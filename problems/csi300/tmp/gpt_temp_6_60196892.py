import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate Multi-Timeframe Momentum Reversal Acceleration
    # Short-term reversal (5-day)
    data['high_5d'] = data['high'].rolling(window=5, min_periods=3).apply(lambda x: x[:-1].max() if len(x) > 1 else np.nan)
    data['low_5d'] = data['low'].rolling(window=5, min_periods=3).apply(lambda x: x[:-1].min() if len(x) > 1 else np.nan)
    data['reversal_high_5d'] = data['close'] / data['high_5d'] - 1
    data['reversal_low_5d'] = data['close'] / data['low_5d'] - 1
    
    # Medium-term reversal (10-day)
    data['high_10d'] = data['high'].rolling(window=10, min_periods=5).apply(lambda x: x[:-3].max() if len(x) > 3 else np.nan)
    data['low_10d'] = data['low'].rolling(window=10, min_periods=5).apply(lambda x: x[:-3].min() if len(x) > 3 else np.nan)
    data['reversal_high_10d'] = data['close'] / data['high_10d'] - 1
    data['reversal_low_10d'] = data['close'] / data['low_10d'] - 1
    
    # Momentum Reversal Acceleration
    data['accel_short'] = data['reversal_high_5d'] - data['reversal_low_5d']
    data['accel_medium'] = data['reversal_high_10d'] - data['reversal_low_10d']
    data['accel_cross'] = data['reversal_high_5d'] - data['reversal_high_10d']
    
    # Calculate Dynamic Volatility-Efficiency Adjustment
    # True Range Efficiency
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    data['volume_per_unit'] = data['volume'] / data['true_range'].replace(0, np.nan)
    data['efficiency_ratio'] = data['volume'] / (data['true_range'] * data['amount']).replace(0, np.nan)
    
    # Volatility Regime Proxy
    data['hl_range'] = data['high'] / data['low'] - 1
    data['hl_volatility'] = data['hl_range'].rolling(window=20, min_periods=10).std()
    
    data['atr'] = data['true_range'].rolling(window=20, min_periods=10).mean()
    
    # Volume Confirmation Signals
    data['volume_ma_20'] = data['volume'].rolling(window=20, min_periods=10).mean()
    data['abnormal_volume'] = data['volume'] / data['volume_ma_20']
    
    data['volume_amount_efficiency'] = data['volume'] / data['amount'].replace(0, np.nan)
    data['vol_adj_efficiency'] = (data['volume'] / data['true_range'].replace(0, np.nan)) / data['atr'].replace(0, np.nan)
    
    # Combine Reversal Acceleration with Efficiency Signals
    # Volatility-Adjust Reversal Components
    data['accel_short_vol_adj'] = data['accel_short'] / data['hl_volatility'].replace(0, np.nan) * data['vol_adj_efficiency']
    data['accel_medium_vol_adj'] = data['accel_medium'] / data['hl_volatility'].replace(0, np.nan) * data['vol_adj_efficiency']
    data['accel_cross_vol_adj'] = data['accel_cross'] / data['hl_volatility'].replace(0, np.nan) * data['vol_adj_efficiency']
    
    # Incorporate Intraday Price Pressure
    data['intraday_return'] = data['close'] / data['open'] - 1
    data['intraday_efficiency'] = data['intraday_return'] * data['volume_amount_efficiency']
    
    # Apply Volume Confirmation Filtering
    volume_threshold = data['volume_amount_efficiency'].rolling(window=20, min_periods=10).quantile(0.3)
    data['volume_filter'] = np.where(data['volume_amount_efficiency'] > volume_threshold, 1, 0.5)
    
    # Generate Composite Alpha Factor
    # Weighted Reversal Acceleration Combination
    data['weighted_accel'] = (0.6 * data['accel_short_vol_adj'] + 
                             0.3 * data['accel_medium_vol_adj'] + 
                             0.1 * data['accel_cross_vol_adj'])
    
    # Efficiency and Volatility Adjustment
    data['adjusted_accel'] = (data['weighted_accel'] * data['vol_adj_efficiency'] + 
                             data['intraday_efficiency']) * data['abnormal_volume'] * data['volume_filter']
    
    # Final Liquidity Enhancement
    data['amount_ma_20'] = data['amount'].rolling(window=20, min_periods=10).mean()
    data['trade_size_consistency'] = data['amount'] / data['amount_ma_20']
    
    # Final factor
    data['alpha_factor'] = data['adjusted_accel'] * data['trade_size_consistency']
    
    # Clean up intermediate columns
    result = data['alpha_factor'].copy()
    
    return result
