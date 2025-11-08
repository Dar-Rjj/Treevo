import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Volatility Regime Detection
    # True Range calculation
    data['prev_close'] = data['close'].shift(1)
    data['TR'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            np.abs(data['high'] - data['prev_close']),
            np.abs(data['low'] - data['prev_close'])
        )
    )
    
    # Regime Classification
    data['TR_MA_20'] = data['TR'].rolling(window=20, min_periods=1).mean()
    data['volatility_regime'] = np.where(data['TR'] > 1.5 * data['TR_MA_20'], 'High', 'Normal')
    
    # Gap Pattern Analysis
    data['overnight_gap'] = data['open'] / data['prev_close'] - 1
    data['gap_fill_ratio'] = (data['close'] - data['open']) / (data['open'] - data['prev_close'])
    data['gap_fill_ratio'] = np.where(np.abs(data['open'] - data['prev_close']) < 1e-8, 0, data['gap_fill_ratio'])
    
    prev_high_low_range = data['high'].shift(1) - data['low'].shift(1)
    data['gap_size_vs_range'] = np.abs(data['overnight_gap']) / np.where(prev_high_low_range < 1e-8, 1, prev_high_low_range)
    
    # Multi-Timeframe Momentum
    data['short_term_alignment'] = np.sign(data['close'] / data['close'].shift(3) - 1) * np.sign(data['close'] / data['close'].shift(5) - 1)
    data['medium_term_trend'] = data['close'] / data['close'].shift(10) - 1
    data['momentum_persistence'] = (data['close'] - data['open']) * (data['open'] - data['prev_close'])
    
    # Volume Confirmation
    data['volume_ema_10'] = data['volume'].ewm(span=10, adjust=False).mean()
    data['avg_volume_20d'] = data['volume'].rolling(window=20, min_periods=1).mean()
    data['gap_volume_intensity'] = data['volume'] * np.abs(data['overnight_gap']) / np.where(data['avg_volume_20d'] < 1e-8, 1, data['avg_volume_20d'])
    
    # Intraday Efficiency
    high_low_range = data['high'] - data['low']
    data['range_efficiency'] = (data['close'] - data['low']) / np.where(high_low_range < 1e-8, 1, high_low_range)
    
    open_prev_close_diff = np.abs(data['open'] - data['prev_close'])
    data['post_gap_efficiency'] = (data['close'] - data['open']) / np.where(open_prev_close_diff < 1e-8, 1, open_prev_close_diff)
    
    # Regime-Adaptive Factor Construction
    factor_values = []
    
    for i in range(len(data)):
        if pd.isna(data.iloc[i]['prev_close']):
            factor_values.append(0)
            continue
            
        if data.iloc[i]['volatility_regime'] == 'High':
            # High Volatility Regime
            gap_momentum_score = np.abs(data.iloc[i]['overnight_gap']) * data.iloc[i]['medium_term_trend'] * data.iloc[i]['volume_ema_10']
            efficiency_adjustment = data.iloc[i]['post_gap_efficiency'] * data.iloc[i]['range_efficiency']
            combined = gap_momentum_score * efficiency_adjustment * data.iloc[i]['TR']
        else:
            # Normal Volatility Regime
            gap_continuity_score = data.iloc[i]['gap_fill_ratio'] * data.iloc[i]['momentum_persistence'] * data.iloc[i]['gap_volume_intensity']
            trend_confirmation = data.iloc[i]['short_term_alignment'] * data.iloc[i]['medium_term_trend']
            combined = gap_continuity_score * trend_confirmation * data.iloc[i]['range_efficiency']
        
        factor_values.append(combined)
    
    factor_series = pd.Series(factor_values, index=data.index)
    return factor_series
