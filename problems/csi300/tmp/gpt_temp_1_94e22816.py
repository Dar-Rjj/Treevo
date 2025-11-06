import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Calculate basic components
    data['high_low_range'] = data['high'] - data['low']
    data['close_open_range'] = data['close'] - data['open']
    data['efficiency_ratio'] = np.where(data['high_low_range'] != 0, 
                                       data['close_open_range'] / data['high_low_range'], 0)
    
    # Volatility-Efficiency Dynamics
    short_term_corr = []
    medium_term_corr = []
    
    for i in range(len(data)):
        if i >= 9:
            # Short-term correlation (4-day window)
            high_low_short = data['high_low_range'].iloc[i-3:i+1].values
            efficiency_short = data['efficiency_ratio'].iloc[i-3:i+1].values
            if len(high_low_short) >= 2 and np.std(high_low_short) > 0 and np.std(efficiency_short) > 0:
                short_corr = np.corrcoef(high_low_short, efficiency_short)[0,1]
                short_term_corr.append(short_corr if not np.isnan(short_corr) else 0)
            else:
                short_term_corr.append(0)
            
            # Medium-term correlation (9-day window)
            high_low_medium = data['high_low_range'].iloc[i-8:i+1].values
            efficiency_medium = data['efficiency_ratio'].iloc[i-8:i+1].values
            if len(high_low_medium) >= 2 and np.std(high_low_medium) > 0 and np.std(efficiency_medium) > 0:
                medium_corr = np.corrcoef(high_low_medium, efficiency_medium)[0,1]
                medium_term_corr.append(medium_corr if not np.isnan(medium_corr) else 0)
            else:
                medium_term_corr.append(0)
        else:
            short_term_corr.append(0)
            medium_term_corr.append(0)
    
    data['short_term_corr'] = short_term_corr
    data['medium_term_corr'] = medium_term_corr
    data['vol_efficiency_regime_shift'] = data['short_term_corr'] - data['medium_term_corr']
    
    # Intraday Volatility Components
    data['gap_volatility_efficiency'] = np.where(
        np.abs(data['open'] - data['close'].shift(1)) > 0,
        np.abs(data['close'] - data['open']) / np.abs(data['open'] - data['close'].shift(1)),
        0
    )
    data['pure_volatility_efficiency'] = np.where(
        data['high_low_range'] > 0,
        np.abs(data['close'] - data['open']) / data['high_low_range'],
        0
    )
    data['intraday_volatility_position'] = np.where(
        data['high_low_range'] > 0,
        (data['close'] - (data['high'] + data['low']) / 2) / data['high_low_range'],
        0
    )
    
    # Volatility Asymmetry Structure
    upside_vol = []
    downside_vol = []
    
    for i in range(len(data)):
        if i >= 3:
            high_max = data['high'].iloc[i-2:i+1].max()
            low_min = data['low'].iloc[i-2:i+1].min()
            close_ref = data['close'].iloc[i-3]
            
            upside = (high_max - close_ref) / close_ref if close_ref > 0 else 0
            downside = (close_ref - low_min) / close_ref if close_ref > 0 else 0
            
            upside_vol.append(upside)
            downside_vol.append(downside)
        else:
            upside_vol.append(0)
            downside_vol.append(0)
    
    data['upside_volatility'] = upside_vol
    data['downside_volatility'] = downside_vol
    data['volatility_asymmetry_ratio'] = np.where(
        data['downside_volatility'] > 0,
        data['upside_volatility'] / data['downside_volatility'],
        1.0
    )
    
    # Volume-Volatility Interaction
    volume_weighted_vol = []
    volume_vol_momentum = []
    
    for i in range(len(data)):
        if i >= 5:
            # Volume-Weighted Volatility Efficiency (5-day window)
            vol_range = data['high_low_range'].iloc[i-4:i+1].values
            volumes = data['volume'].iloc[i-4:i+1].values
            close_ref = data['close'].iloc[i-5]
            
            if close_ref > 0 and np.sum(volumes) > 0:
                weighted_vol = np.sum((vol_range / close_ref) * volumes) / np.sum(volumes)
                volume_weighted_vol.append(weighted_vol)
            else:
                volume_weighted_vol.append(0)
            
            # Volume-Volatility Momentum
            if i >= 3:
                current_ratio = data['volume'].iloc[i] / data['high_low_range'].iloc[i] if data['high_low_range'].iloc[i] > 0 else 0
                past_ratio = data['volume'].iloc[i-3] / data['high_low_range'].iloc[i-3] if data['high_low_range'].iloc[i-3] > 0 else 0
                
                if past_ratio > 0:
                    momentum = (current_ratio / past_ratio) - 1
                    volume_vol_momentum.append(momentum)
                else:
                    volume_vol_momentum.append(0)
            else:
                volume_vol_momentum.append(0)
        else:
            volume_weighted_vol.append(0)
            volume_vol_momentum.append(0)
    
    data['volume_weighted_volatility'] = volume_weighted_vol
    data['volume_volatility_momentum'] = volume_vol_momentum
    data['volatility_volume_consistency'] = np.sign(data['high_low_range']) * np.sign(data['volume_weighted_volatility'])
    
    # Regime Classification & Signal Construction
    regime_signals = []
    
    for i in range(len(data)):
        if i >= 9:  # Ensure we have enough data for calculations
            asym_ratio = data['volatility_asymmetry_ratio'].iloc[i]
            pure_eff = data['pure_volatility_efficiency'].iloc[i]
            regime_shift = data['vol_efficiency_regime_shift'].iloc[i]
            gap_eff = data['gap_volatility_efficiency'].iloc[i]
            vol_pos = data['intraday_volatility_position'].iloc[i]
            vol_mom = data['volume_volatility_momentum'].iloc[i]
            vol_cons = data['volatility_volume_consistency'].iloc[i]
            
            # Determine regimes
            if asym_ratio > 1.4:
                asym_regime = 'high'
            elif asym_ratio >= 0.8:
                asym_regime = 'normal'
            else:
                asym_regime = 'low'
            
            eff_regime = 'high' if pure_eff > 0.6 else 'low'
            
            # Construct regime-adaptive signals
            if asym_regime == 'high' and eff_regime == 'high':
                signal = -gap_eff * vol_cons * np.abs(regime_shift)
            elif asym_regime == 'normal' and eff_regime == 'low':
                signal = pure_eff * vol_mom * np.abs(regime_shift)
            elif asym_regime == 'low' and eff_regime == 'high':
                signal = regime_shift * np.abs(vol_pos) * vol_mom
            else:
                signal = 0
                
            regime_signals.append(signal)
        else:
            regime_signals.append(0)
    
    data['regime_signal'] = regime_signals
    
    # Final Alpha Construction
    data['volatility_asymmetry_scaling'] = data['regime_signal'] / (data['volatility_asymmetry_ratio'] + 0.0001)
    data['volatility_volume_alignment'] = np.sign(data['high_low_range']) * np.sign(data['volume_weighted_volatility'])
    
    # Final Alpha
    alpha = data['volatility_asymmetry_scaling'] * (1 + 0.1 * data['volatility_volume_alignment'])
    
    return alpha
