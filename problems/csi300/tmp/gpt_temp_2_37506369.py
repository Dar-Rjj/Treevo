import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Gap Efficiency Divergence
    # Short-Term Gap Efficiency
    data['short_gap_eff'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'])
    data['short_gap_eff'] = data['short_gap_eff'].replace([np.inf, -np.inf], np.nan)
    
    # Medium-Term Gap Efficiency
    data['medium_gap_eff'] = np.abs(data['close'] - data['open'].shift(5)) / (
        (data['high'] - data['low']).rolling(window=6).sum()
    )
    data['medium_gap_eff'] = data['medium_gap_eff'].replace([np.inf, -np.inf], np.nan)
    
    # Gap Efficiency Divergence
    data['gap_eff_div'] = data['short_gap_eff'] - data['medium_gap_eff']
    
    # Turnover-Momentum Divergence
    # Momentum Difference
    data['momentum_diff'] = (data['close'] / data['close'].shift(4) - 1) - (data['close'] / data['close'].shift(9) - 1)
    
    # Turnover Ratio
    data['dollar_vol'] = data['volume'] * data['close']
    data['turnover_ratio'] = (
        data['dollar_vol'].rolling(window=5).mean() / 
        data['dollar_vol'].rolling(window=15).mean() - 1
    )
    
    # Turnover-Momentum Divergence
    data['turnover_momentum_div'] = data['momentum_diff'] * data['turnover_ratio']
    
    # Combine Gap-Momentum Divergence
    data['gap_momentum_div'] = np.sqrt(
        np.abs(data['gap_eff_div'] * data['turnover_momentum_div']) * 
        np.sign(data['gap_eff_div'] * data['turnover_momentum_div'])
    )
    
    # Volume-Pressure Confirmation System
    # Pressure Asymmetry
    data['morning_gap_pressure'] = (data['high'] - data['open']) / np.abs(data['open'] - data['close'].shift(1))
    data['morning_gap_pressure'] = data['morning_gap_pressure'].replace([np.inf, -np.inf], np.nan)
    
    data['gap_fill_pressure'] = (data['close'] - data['open']) / np.abs(data['open'] - data['close'].shift(1))
    data['gap_fill_pressure'] = data['gap_fill_pressure'].replace([np.inf, -np.inf], np.nan)
    
    data['pressure_asymmetry'] = data['morning_gap_pressure'] - data['gap_fill_pressure']
    
    # Volume Asymmetry
    # Upside Volume Ratio
    data['returns'] = data['close'].pct_change()
    data['is_up_day'] = data['returns'] > 0
    
    up_volume_avg = []
    for i in range(len(data)):
        if i >= 9:
            window_data = data.iloc[i-9:i+1]
            up_days = window_data[window_data['is_up_day']]
            if len(up_days) > 0:
                up_vol_ratio = up_days['volume'].mean() / window_data['volume'].mean()
            else:
                up_vol_ratio = 0
        else:
            up_vol_ratio = np.nan
        up_volume_avg.append(up_vol_ratio)
    
    data['upside_volume_ratio'] = up_volume_avg
    
    # Price Asymmetry
    pos_returns_sum = []
    neg_returns_sum = []
    for i in range(len(data)):
        if i >= 9:
            window_data = data.iloc[i-9:i+1]
            pos_returns = window_data['returns'][window_data['returns'] > 0].sum()
            neg_returns = window_data['returns'][window_data['returns'] < 0].sum()
        else:
            pos_returns = np.nan
            neg_returns = np.nan
        pos_returns_sum.append(pos_returns)
        neg_returns_sum.append(neg_returns)
    
    data['pos_returns_sum'] = pos_returns_sum
    data['neg_returns_sum'] = neg_returns_sum
    data['price_asymmetry'] = np.log(1 + data['pos_returns_sum']) - np.log(1 + np.abs(data['neg_returns_sum']))
    
    # Volume Asymmetry
    data['volume_asymmetry'] = data['upside_volume_ratio'] * data['price_asymmetry']
    
    # Combine Volume-Pressure Confirmation
    data['volume_pressure_conf'] = np.cbrt(
        np.abs(data['pressure_asymmetry'] * data['volume_asymmetry']) * 
        np.sign(data['pressure_asymmetry'] * data['volume_asymmetry'])
    )
    
    # Final Alpha Synthesis
    # Base Signal
    data['base_signal'] = data['gap_momentum_div'] * data['volume_pressure_conf']
    
    # Risk Adjustment (True Range)
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            np.abs(data['high'] - data['close'].shift(1)),
            np.abs(data['low'] - data['close'].shift(1))
        )
    )
    data['avg_true_range'] = data['true_range'].rolling(window=20).mean()
    
    data['risk_adjusted'] = data['base_signal'] / data['avg_true_range']
    
    # Volatility Scaling
    data['abs_intraday_move'] = np.abs(data['close'] - data['open'])
    data['volatility_scale'] = (
        data['abs_intraday_move'].rolling(window=5).sum() / 
        data['abs_intraday_move'].rolling(window=10).sum() - 1
    )
    
    # Final Alpha Factor
    data['alpha_factor'] = data['risk_adjusted'] * data['volatility_scale']
    
    return data['alpha_factor']
