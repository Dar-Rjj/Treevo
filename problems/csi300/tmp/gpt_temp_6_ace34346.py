import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Calculate Price Acceleration Components
    # Raw Price Momentum
    data['ret_1d'] = data['close'] / data['close'].shift(1) - 1
    data['mom_3d'] = data['close'] / data['close'].shift(3) - 1
    data['mom_5d'] = data['close'] / data['close'].shift(5) - 1
    
    # Acceleration Metrics
    data['accel_3d'] = (data['close'] / data['close'].shift(3)) / (data['close'].shift(3) / data['close'].shift(6)) - 1
    data['accel_5d'] = (data['close'] / data['close'].shift(5)) / (data['close'].shift(5) / data['close'].shift(10)) - 1
    
    # Acceleration persistence
    data['accel_3d_pos'] = (data['accel_3d'] > 0).astype(int)
    data['accel_persistence'] = data['accel_3d_pos'] * (data['accel_3d_pos'].groupby((data['accel_3d_pos'] != data['accel_3d_pos'].shift(1)).cumsum()).cumcount() + 1)
    
    # Analyze Volume-Price Asymmetry
    # Volume Expansion Patterns
    data['volume_ratio'] = data['volume'] / data['volume'].shift(1)
    data['volume_3d_avg'] = data['volume'].rolling(window=3, min_periods=1).mean().shift(1)
    data['volume_mom_3d'] = data['volume'] / data['volume_3d_avg']
    data['volume_accel'] = (data['volume'] / data['volume'].shift(1)) / (data['volume'].shift(1) / data['volume'].shift(2))
    
    # Asymmetric Response Detection
    vol_avg = data['volume'].rolling(window=20, min_periods=1).mean()
    data['up_vol_intensity'] = np.where(data['ret_1d'] > 0, data['volume'] / vol_avg, np.nan)
    data['down_vol_intensity'] = np.where(data['ret_1d'] < 0, data['volume'] / vol_avg, np.nan)
    data['vol_price_divergence'] = np.sign(data['ret_1d']) * (data['volume'] - vol_avg)
    
    # Compute Momentum Decay Characteristics
    # Momentum Half-Life Estimation
    data['ret_5d'] = data['close'] / data['close'].shift(5) - 1
    autocorr = data['ret_5d'].rolling(window=30, min_periods=10).apply(lambda x: x.autocorr(lag=1), raw=False)
    data['decay_rate'] = 1 - autocorr
    data['half_life'] = np.log(0.5) / np.log(np.clip(1 - data['decay_rate'], 1e-6, 0.999))
    
    # Decay-Adjusted Momentum
    decay_param = 1 - np.exp(-np.log(2) / np.clip(data['half_life'], 1, 30))
    weights = [decay_param.iloc[i] * ((1 - decay_param.iloc[i]) ** j) for i in range(len(data)) for j in range(5)]
    weight_matrix = np.array(weights).reshape(len(data), 5)
    past_returns = pd.DataFrame({f'ret_lag_{i}': data['ret_1d'].shift(i) for i in range(1, 6)})
    data['decay_adj_mom'] = (weight_matrix * past_returns.values).sum(axis=1) / weight_matrix.sum(axis=1)
    
    # Assess Intraday Price Efficiency
    # Capture Efficiency
    daily_range = data['high'] - data['low']
    data['daily_capture'] = (data['close'] - data['open']) / np.where(daily_range == 0, 1e-6, daily_range)
    data['capture_eff_5d'] = data['daily_capture'].rolling(window=5, min_periods=1).mean()
    
    # Reversal Tendency
    data['intraday_reversal'] = (data['close'] - data['open']) * (data['high'] - data['close'])
    data['gap_closure_eff'] = np.abs(data['open'] - data['close'].shift(1)) / np.where(daily_range == 0, 1e-6, daily_range)
    mid_price = (data['high'] + data['low']) / 2
    data['eod_momentum'] = (data['close'] - mid_price) / np.where(daily_range == 0, 1e-6, daily_range)
    
    # Synthesize Composite Acceleration Factor
    # Combine Acceleration with Volume Confirmation
    volume_expansion = data['volume_mom_3d'] * data['volume_accel']
    asymmetric_weight = np.where(data['ret_1d'] > 0, 1.2, 0.8)
    accel_volume_combo = data['accel_5d'] * volume_expansion * asymmetric_weight
    accel_volume_combo = accel_volume_combo - 0.1 * data['vol_price_divergence']
    
    # Apply Decay Adjustment
    half_life_adj = np.where(data['half_life'] > 10, 1.5, 
                            np.where(data['half_life'] > 5, 1.2, 1.0))
    decay_adj_factor = accel_volume_combo * half_life_adj * data['decay_adj_mom']
    
    # Integrate Intraday Efficiency
    reversal_adj = np.where(data['intraday_reversal'] > 0, 0.8, 1.2)
    eod_boost = 1 + 0.3 * np.abs(data['eod_momentum'])
    final_factor = decay_adj_factor * data['capture_eff_5d'] * reversal_adj * eod_boost
    
    return final_factor
