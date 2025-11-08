import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility Regime Identification
    # True Range calculation
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # High Volatility Regime
    data['tr_ma_20'] = data['true_range'].rolling(window=20, min_periods=1).mean()
    data['high_vol_regime'] = (data['true_range'] > 1.5 * data['tr_ma_20']).astype(int)
    
    # Intraday Reversal Dynamics
    # Gap Reversal Strength
    data['opening_gap_magnitude'] = (data['open'] - data['prev_close']) / data['prev_close']
    
    # Intraday Reversal Ratio
    gap_sign = np.sign(data['open'] - data['prev_close'])
    reversal_sign = np.sign(data['close'] - data['open'])
    reversal_condition = (gap_sign != reversal_sign) & (gap_sign != 0)
    data['intraday_reversal_ratio'] = 0
    data.loc[reversal_condition, 'intraday_reversal_ratio'] = (
        (data['close'] - data['open']) / (data['open'] - data['prev_close'])
    )
    
    # Gap Fill Efficiency
    data['gap_fill_efficiency'] = abs(data['close'] - data['prev_close']) / abs(data['open'] - data['prev_close'])
    data['gap_fill_efficiency'] = data['gap_fill_efficiency'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Extreme Price Rejection
    data['high_rejection_signal'] = (data['high'] - data['close']) / (data['high'] - data['low'])
    data['low_rejection_signal'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    data['rejection_asymmetry'] = data['high_rejection_signal'] - data['low_rejection_signal']
    
    # Volume at Price Extremes
    data['high_rejection_volume'] = data['volume'] * data['high_rejection_signal']
    data['low_rejection_volume'] = data['volume'] * data['low_rejection_signal']
    data['rejection_volume_divergence'] = data['high_rejection_volume'] - data['low_rejection_volume']
    
    # Momentum-Range Divergence Structure
    # Price-Volume Divergence
    data['prev_close_2'] = data['close'].shift(2)
    data['prev_volume_2'] = data['volume'].shift(2)
    data['prev_close_5'] = data['close'].shift(5)
    data['prev_volume_5'] = data['volume'].shift(5)
    
    data['short_term_divergence'] = (
        np.sign(data['close'] - data['prev_close_2']) * 
        np.sign(data['volume'] - data['prev_volume_2'])
    )
    data['medium_term_divergence'] = (
        np.sign(data['close'] - data['prev_close_5']) * 
        np.sign(data['volume'] - data['prev_volume_5'])
    )
    
    # Divergence Persistence
    divergence_signs = []
    for i in range(len(data)):
        if i < 3:
            divergence_signs.append(0)
        else:
            recent_divergence = data['short_term_divergence'].iloc[i-2:i+1]
            consistent_count = sum(recent_divergence == recent_divergence.iloc[-1])
            divergence_signs.append(consistent_count)
    data['divergence_persistence'] = divergence_signs
    
    # Range Pressure Analysis
    data['prev_high_1'] = data['high'].shift(1)
    data['prev_low_1'] = data['low'].shift(1)
    data['prev_high_2'] = data['high'].shift(2)
    data['prev_low_2'] = data['low'].shift(2)
    data['prev_high_3'] = data['high'].shift(3)
    data['prev_low_3'] = data['low'].shift(3)
    
    data['range_acceleration'] = (
        (data['high'] - data['low']) - 
        2 * (data['prev_high_1'] - data['prev_low_1']) + 
        (data['prev_high_2'] - data['prev_low_2'])
    )
    
    data['range_vs_historical'] = (
        (data['high'] - data['low']) / 
        ((data['prev_high_1'] - data['prev_low_1'] + 
          data['prev_high_2'] - data['prev_low_2'] + 
          data['prev_high_3'] - data['prev_low_3']) / 3)
    )
    data['range_vs_historical'] = data['range_vs_historical'].replace([np.inf, -np.inf], 1).fillna(1)
    
    data['opening_gap_pressure'] = abs(data['open'] - data['prev_close']) / (data['high'] - data['low'])
    data['opening_gap_pressure'] = data['opening_gap_pressure'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Momentum Deceleration
    data['prev_close_1'] = data['close'].shift(1)
    data['prev_volume_1'] = data['volume'].shift(1)
    
    data['price_deceleration'] = (
        (data['close'] - data['prev_close_1']) - 
        (data['prev_close_1'] - data['prev_close_2'])
    )
    data['volume_deceleration'] = (
        (data['volume'] - data['prev_volume_1']) - 
        (data['prev_volume_1'] - data['prev_volume_2'])
    )
    data['deceleration_correlation'] = data['price_deceleration'] * data['volume_deceleration']
    
    # Efficiency Convergence Dynamics
    # Intraday Efficiency Analysis
    data['current_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['current_efficiency'] = data['current_efficiency'].replace([np.inf, -np.inf], 0).fillna(0)
    
    data['prev_open_5'] = data['open'].shift(5)
    data['prev_high_5'] = data['high'].shift(5)
    data['prev_low_5'] = data['low'].shift(5)
    data['prev_close_5'] = data['close'].shift(5)
    
    data['historical_efficiency'] = (
        (data['prev_close_5'] - data['prev_open_5']) / 
        (data['prev_high_5'] - data['prev_low_5'])
    )
    data['historical_efficiency'] = data['historical_efficiency'].replace([np.inf, -np.inf], 0).fillna(0)
    
    data['efficiency_momentum'] = data['current_efficiency'] - data['historical_efficiency']
    
    # Volume Confirmation Patterns
    data['volume_trend'] = data['volume'].rolling(window=10, min_periods=1).mean()
    
    data['prev_volume_10'] = data['volume'].shift(10)
    data['volume_range_alignment'] = (
        (data['volume'] - 2 * data['prev_volume_5'] + data['prev_volume_10']) * 
        data['range_acceleration']
    )
    
    data['closing_volume_intensity'] = (
        data['volume'] * abs(data['close'] - data['open']) / data['open']
    )
    
    # Reversal-Volume Convergence
    # Gap Fill Quality
    data['gap_fill_quality'] = data['gap_fill_efficiency'] * data['rejection_volume_divergence']
    
    # Alignment Score
    data['alignment_score'] = data['divergence_persistence'] * data['rejection_asymmetry']
    
    # Final Alpha Synthesis
    # Regime-Adaptive Core Composite
    data['core_composite'] = (
        data['gap_fill_quality'] * data['divergence_persistence'] * 
        data['range_vs_historical'] * data['efficiency_momentum']
    )
    
    # Volume-Pressure Convergence Layer
    data['volume_pressure_layer'] = (
        data['rejection_volume_divergence'] * 
        data['volume_range_alignment'] * 
        data['closing_volume_intensity']
    )
    
    # Regime-Adaptive Weighting
    regime_multiplier = np.where(data['high_vol_regime'] == 1, 1.5, 1.0)
    
    # Final Alpha Factor
    alpha_factor = (
        data['core_composite'] * 
        data['volume_pressure_layer'] * 
        regime_multiplier
    )
    
    return alpha_factor
