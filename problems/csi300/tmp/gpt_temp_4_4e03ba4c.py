import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate previous close
    data['prev_close'] = data['close'].shift(1)
    
    # Calculate basic volatility measures
    data['daily_range'] = data['high'] - data['low']
    data['prev_range'] = data['daily_range'].shift(1)
    
    # Calculate volatility regime
    vol_ma = data['daily_range'].rolling(window=20, min_periods=10).mean()
    vol_std = data['daily_range'].rolling(window=20, min_periods=10).std()
    data['vol_regime'] = np.where(
        data['daily_range'] > vol_ma + vol_std, 'high',
        np.where(data['daily_range'] < vol_ma - vol_std, 'low', 'normal')
    )
    
    # Initialize factor components
    regime_flow = pd.Series(index=data.index, dtype=float)
    price_volume_div = pd.Series(index=data.index, dtype=float)
    vol_convexity = pd.Series(index=data.index, dtype=float)
    cross_timeframe = pd.Series(index=data.index, dtype=float)
    regime_efficiency = pd.Series(index=data.index, dtype=float)
    
    # 1. Regime-Adaptive Order Flow
    ma_range_5 = data['daily_range'].rolling(window=5, min_periods=3).mean()
    ma_volume_5 = data['volume'].rolling(window=5, min_periods=3).mean()
    
    for i in range(len(data)):
        if i < 1:
            continue
            
        if data['vol_regime'].iloc[i] == 'high':
            regime_flow.iloc[i] = (data['open'].iloc[i] - data['prev_close'].iloc[i]) / max(data['daily_range'].iloc[i], 1e-6)
        elif data['vol_regime'].iloc[i] == 'low':
            if i >= 5 and not pd.isna(ma_range_5.iloc[i]):
                regime_flow.iloc[i] = (data['close'].iloc[i] - data['open'].iloc[i]) / max(ma_range_5.iloc[i], 1e-6)
        else:  # normal
            if i >= 5 and not pd.isna(ma_volume_5.iloc[i]):
                regime_flow.iloc[i] = (data['volume'].iloc[i] / ma_volume_5.iloc[i]) * (data['close'].iloc[i] - data['open'].iloc[i]) / max(data['daily_range'].iloc[i], 1e-6)
    
    # 2. Multi-Scale Price-Volume Divergence
    ma_close_3 = data['close'].rolling(window=3, min_periods=2).mean()
    ma_close_8 = data['close'].rolling(window=8, min_periods=5).mean()
    ma_close_21 = data['close'].rolling(window=21, min_periods=15).mean()
    ma_volume_3 = data['volume'].rolling(window=3, min_periods=2).mean()
    ma_volume_8 = data['volume'].rolling(window=8, min_periods=5).mean()
    ma_volume_21 = data['volume'].rolling(window=21, min_periods=15).mean()
    
    for i in range(len(data)):
        if i >= 8:
            short_term = (ma_close_3.iloc[i] / ma_close_8.iloc[i]) / max(ma_volume_3.iloc[i] / ma_volume_8.iloc[i], 1e-6)
            medium_term = (ma_close_8.iloc[i] / ma_close_21.iloc[i]) / max(ma_volume_8.iloc[i] / ma_volume_21.iloc[i], 1e-6)
        else:
            short_term, medium_term = np.nan, np.nan
            
        if i >= 20:
            long_term = (data['close'].iloc[i] / ma_close_21.iloc[i]) / max(data['volume'].iloc[i] / ma_volume_21.iloc[i], 1e-6)
        else:
            long_term = np.nan
            
        if not (pd.isna(short_term) or pd.isna(medium_term) or pd.isna(long_term)):
            price_volume_div.iloc[i] = (short_term + medium_term + long_term) / 3
    
    # 3. Volatility-Convexity Alignment
    ma_close_5 = data['close'].rolling(window=5, min_periods=3).mean()
    
    for i in range(len(data)):
        if data['vol_regime'].iloc[i] == 'high':
            vol_convexity.iloc[i] = ((data['high'].iloc[i] - data['close'].iloc[i]) + (data['close'].iloc[i] - data['low'].iloc[i])) / max(data['daily_range'].iloc[i], 1e-6)
        elif data['vol_regime'].iloc[i] == 'low':
            if i >= 5 and not pd.isna(ma_close_5.iloc[i]):
                vol_convexity.iloc[i] = (data['close'].iloc[i] - ma_close_5.iloc[i]) / max(data['daily_range'].iloc[i], 1e-6)
        else:  # normal
            if i >= 5 and not pd.isna(ma_volume_5.iloc[i]):
                vol_convexity.iloc[i] = (data['close'].iloc[i] - data['open'].iloc[i]) / max(data['daily_range'].iloc[i], 1e-6) * data['volume'].iloc[i] / ma_volume_5.iloc[i]
    
    # 4. Cross-Timeframe Pressure
    cross_pressure = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if i < 1:
            continue
            
        denominator = max(
            data['daily_range'].iloc[i],
            abs(data['high'].iloc[i] - data['prev_close'].iloc[i]),
            abs(data['low'].iloc[i] - data['prev_close'].iloc[i]),
            1e-6
        )
        cross_pressure.iloc[i] = (data['open'].iloc[i] - data['prev_close'].iloc[i]) / denominator
    
    # Medium-term: 5-day MA
    cross_pressure_ma_5 = cross_pressure.rolling(window=5, min_periods=3).mean()
    
    # Long-term: current vs 5 days ago
    cross_pressure_shift_5 = cross_pressure.shift(5)
    
    for i in range(len(data)):
        if i < 1:
            continue
            
        short_term_val = cross_pressure.iloc[i]
        
        if i >= 5 and not pd.isna(cross_pressure_ma_5.iloc[i]):
            medium_term_val = cross_pressure_ma_5.iloc[i]
        else:
            medium_term_val = np.nan
            
        if i >= 6 and not pd.isna(cross_pressure_shift_5.iloc[i]) and abs(cross_pressure_shift_5.iloc[i]) > 1e-6:
            long_term_val = cross_pressure.iloc[i] / cross_pressure_shift_5.iloc[i]
        else:
            long_term_val = np.nan
            
        if not (pd.isna(short_term_val) or pd.isna(medium_term_val) or pd.isna(long_term_val)):
            cross_timeframe.iloc[i] = (short_term_val + medium_term_val + long_term_val) / 3
    
    # 5. Regime-Efficiency Signals
    # Calculate price change percentage
    data['price_change_pct'] = abs((data['close'] - data['prev_close']) / data['prev_close'])
    
    # Count large moves in last 10 days
    large_move_count = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 10:
            window_data = data['price_change_pct'].iloc[i-9:i+1]
            large_move_count.iloc[i] = (window_data > 0.01).sum()
        else:
            large_move_count.iloc[i] = np.nan
    
    for i in range(len(data)):
        if data['vol_regime'].iloc[i] == 'high':
            regime_efficiency.iloc[i] = data['volume'].iloc[i] / max(data['daily_range'].iloc[i], 1e-6)
        elif data['vol_regime'].iloc[i] == 'low':
            if i >= 1:
                denominator = max(
                    data['daily_range'].iloc[i],
                    abs(data['high'].iloc[i] - data['prev_close'].iloc[i]),
                    abs(data['low'].iloc[i] - data['prev_close'].iloc[i]),
                    1e-6
                )
                regime_efficiency.iloc[i] = (data['close'].iloc[i] - data['open'].iloc[i]) / denominator
        else:  # normal
            if i >= 10 and not pd.isna(large_move_count.iloc[i]) and not pd.isna(ma_volume_5.iloc[i]):
                regime_efficiency.iloc[i] = (large_move_count.iloc[i] / 10) * data['volume'].iloc[i] / ma_volume_5.iloc[i]
    
    # Combine all components with equal weighting
    factor = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        components = []
        if not pd.isna(regime_flow.iloc[i]):
            components.append(regime_flow.iloc[i])
        if not pd.isna(price_volume_div.iloc[i]):
            components.append(price_volume_div.iloc[i])
        if not pd.isna(vol_convexity.iloc[i]):
            components.append(vol_convexity.iloc[i])
        if not pd.isna(cross_timeframe.iloc[i]):
            components.append(cross_timeframe.iloc[i])
        if not pd.isna(regime_efficiency.iloc[i]):
            components.append(regime_efficiency.iloc[i])
        
        if components:
            factor.iloc[i] = np.mean(components)
    
    return factor
