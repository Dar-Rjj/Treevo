import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Volatility Regime Classification
    data['micro_vol'] = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1) + 0.0001)
    
    # Meso volatility (5-day vs 5-day)
    data['meso_vol_num'] = data['high'].rolling(window=5).apply(lambda x: (x - data.loc[x.index, 'low']).sum(), raw=False)
    data['meso_vol_den'] = data['high'].shift(5).rolling(window=5).apply(lambda x: (x - data.loc[x.index, 'low'].shift(5)).sum(), raw=False)
    data['meso_vol'] = data['meso_vol_num'] / (data['meso_vol_den'] + 0.0001)
    
    # Macro volatility (20-day vs 20-day)
    data['macro_vol_num'] = data['high'].rolling(window=20).max() - data['low'].rolling(window=20).min()
    data['macro_vol_den'] = data['high'].shift(20).rolling(window=20).max() - data['low'].shift(20).rolling(window=20).min()
    data['macro_vol'] = data['macro_vol_num'] / (data['macro_vol_den'] + 0.0001)
    
    # Volatility-Weighted Entropy Divergence
    data['vol_adj_price_entropy'] = ((data['high'] - data['low']) / (data['close'] - data['open'] + 0.0001)) * data['micro_vol']
    data['volume_vol_entropy'] = (data['volume'] / (data['volume'] - data['volume'].shift(1) + 0.0001)) * (data['volume'] / (data['volume'].shift(1) + 0.0001))
    data['vol_entropy_div_signal'] = data['vol_adj_price_entropy'] * data['volume_vol_entropy']
    
    # Gap-Enhanced Fractal Asymmetry
    data['vol_gap_fractal_pressure'] = ((data['open'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1) + 0.0001)) * ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'] + 0.0001)
    data['gap_volume_entropy_conf'] = data['vol_gap_fractal_pressure'] * data['volume_vol_entropy']
    data['enhanced_fractal_asym'] = data['gap_volume_entropy_conf'] * np.sign(data['close'] - data['open'])
    
    # Multi-Fractal Entropy Patterns
    data['micro_fractal_entropy'] = ((data['close'] - (data['high'] + data['low'])/2) / (data['high'] - data['low'] + 0.0001)) * data['vol_adj_price_entropy']
    data['macro_fractal_entropy'] = ((data['close'] - (data['high'].shift(3) + data['low'].shift(3))/2) / (data['high'].shift(3) - data['low'].shift(3) + 0.0001)) * data['volume_vol_entropy']
    data['fractal_entropy_alignment'] = data['micro_fractal_entropy'] * data['macro_fractal_entropy']
    
    # Price Momentum Structure
    data['micro_momentum'] = (data['close'] - data['close'].shift(1)) / (data['high'] - data['low'] + 0.0001)
    data['meso_momentum'] = (data['close'] - data['close'].shift(5)) / (data['high'].rolling(window=6).max() - data['low'].rolling(window=6).min() + 0.0001)
    data['macro_momentum'] = (data['close'] - data['close'].shift(20)) / (data['high'].rolling(window=21).max() - data['low'].rolling(window=21).min() + 0.0001)
    
    # Entropy-Momentum Coupling
    data['vol_weighted_entropy_momentum'] = data['micro_fractal_entropy'] * data['vol_adj_price_entropy'] * data['micro_vol']
    data['volume_entropy_fractal_momentum'] = data['micro_fractal_entropy'] * data['volume_vol_entropy'] * (data['volume'] / (data['volume'].shift(1) + 0.0001))
    data['entropy_momentum_convergence'] = data['vol_weighted_entropy_momentum'] * data['volume_entropy_fractal_momentum']
    
    # Volume Flow Dynamics
    data['volume_acceleration'] = (data['volume'] / (data['volume'].shift(1) + 0.0001)) - (data['volume'].shift(1) / (data['volume'].shift(2) + 0.0001))
    
    def volume_persistence_func(x):
        return sum(x[i] > x[i-1] for i in range(1, len(x))) / 5 if len(x) == 5 else 0
    
    data['volume_persistence'] = data['volume'].rolling(window=5).apply(volume_persistence_func, raw=True)
    data['volume_momentum_correlation'] = np.sign(data['close'] - data['close'].shift(1)) * np.sign(data['volume'] - data['volume'].shift(1))
    
    # Amount-Volume Entropy Integration
    data['vol_adj_amount_entropy'] = ((data['close'] - data['open']) * (data['amount'] / (data['volume'] + 0.0001)) / (data['high'] - data['low'] + 0.0001)) * data['vol_adj_price_entropy']
    data['entropy_amount_momentum'] = data['vol_adj_amount_entropy'] * (data['amount'] / (data['amount'].shift(1) + 0.0001)) * data['volume_vol_entropy']
    data['amount_entropy_signal'] = data['entropy_amount_momentum'] * data['micro_vol']
    
    # Integrated Volume Microstructure
    data['volume_entropy_resonance'] = data['volume_vol_entropy'] * data['fractal_entropy_alignment']
    data['price_volatility_resonance'] = data['vol_adj_price_entropy'] * data['micro_fractal_entropy']
    data['multi_fractal_volume_micro'] = data['volume_entropy_resonance'] * data['price_volatility_resonance']
    
    # Gap Dynamics Analysis
    data['opening_gap_momentum'] = ((data['open'] - data['close'].shift(1)) / (data['close'].shift(1) + 0.0001)) * np.sign(data['close'] - data['open'])
    data['intraday_range_efficiency'] = abs(data['close'] - data['open']) / (data['high'] - data['low'] + 0.0001)
    data['gap_close_efficiency'] = abs(data['close'] - data['close'].shift(1)) / (abs(data['open'] - data['close'].shift(1)) + 0.0001)
    
    # Multi-Scale Momentum Divergence
    data['micro_meso_divergence'] = data['micro_momentum'] - data['meso_momentum']
    data['meso_macro_divergence'] = data['meso_momentum'] - data['macro_momentum']
    data['vol_momentum_divergence'] = (data['micro_vol'] - data['meso_vol']) * (data['micro_momentum'] - data['macro_momentum'])
    
    # Fractal Gap Dynamics
    data['gap_entropy_integration'] = data['opening_gap_momentum'] * data['vol_entropy_div_signal']
    data['efficiency_entropy_alignment'] = data['intraday_range_efficiency'] * data['enhanced_fractal_asym']
    data['multi_scale_gap_entropy'] = data['gap_entropy_integration'] * data['efficiency_entropy_alignment']
    
    # Momentum Acceleration Framework
    data['micro_acceleration'] = data['micro_momentum'] - ((data['close'].shift(1) - data['close'].shift(2)) / (data['high'].shift(1) - data['low'].shift(1) + 0.0001))
    
    def meso_acceleration_func(x):
        idx = x.index
        if len(idx) < 6:
            return 0
        close_diff = data.loc[idx[5], 'close'] - data.loc[idx[0], 'close']
        price_range = data.loc[idx[0:5], 'high'].max() - data.loc[idx[0:5], 'low'].min()
        return close_diff / (price_range + 0.0001)
    
    data['meso_acceleration'] = data['close'].rolling(window=6).apply(meso_acceleration_func, raw=False)
    data['acceleration_divergence'] = data['micro_acceleration'] * data['meso_acceleration']
    
    # Fractal Momentum Intensity
    data['price_entropy'] = (data['high'] - data['low']) / (data['close'] - data['open'] + 0.0001)
    data['entropy_breakout'] = (data['price_entropy'] / data['price_entropy'].rolling(window=10).mean()) - 1
    data['fractal_momentum_intensity'] = abs(data['micro_fractal_entropy']) + abs(data['macro_fractal_entropy'])
    
    # Entropy-Momentum Acceleration
    data['vol_entropy_acceleration'] = data['vol_weighted_entropy_momentum'] * data['acceleration_divergence']
    data['volume_entropy_acceleration'] = data['volume_entropy_fractal_momentum'] * data['fractal_momentum_intensity']
    data['multi_scale_entropy_acceleration'] = data['vol_entropy_acceleration'] * data['volume_entropy_acceleration']
    
    # Breakout Confirmation System
    data['range_breakout'] = (data['close'] - data['low'].rolling(window=10).min()) / (data['high'].rolling(window=10).max() - data['low'].rolling(window=10).min() + 0.0001)
    data['volume_breakout'] = data['volume'] / data['volume'].rolling(window=10).max()
    
    data['entropy_breakout_threshold'] = (abs(data['entropy_breakout']) > 0.1).astype(float)
    data['fractal_regime_filter'] = (data['fractal_momentum_intensity'] > 0.5).astype(float)
    data['breakout_entropy_signal'] = data['entropy_breakout_threshold'] * data['fractal_regime_filter']
    
    data['price_volume_breakout'] = data['range_breakout'] * data['volume_breakout'] * np.sign(data['micro_momentum'])
    data['entropy_volatility_breakout'] = data['breakout_entropy_signal'] * data['vol_entropy_div_signal']
    data['integrated_breakout_confidence'] = data['price_volume_breakout'] * data['entropy_volatility_breakout']
    
    # Regime Detection and Alpha Construction
    high_vol_condition = (data['micro_vol'] > 1.5) & (data['meso_vol'] > 1.2) & (data['vol_entropy_div_signal'] > 0)
    low_vol_condition = (data['micro_vol'] < 0.7) & (data['meso_vol'] < 0.8) & (data['vol_entropy_div_signal'] < 0)
    expanding_vol_condition = (data['micro_vol'] > data['meso_vol']) & (data['meso_vol'] > data['macro_vol']) & (data['enhanced_fractal_asym'] > 0)
    contracting_vol_condition = (data['micro_vol'] < data['meso_vol']) & (data['meso_vol'] < data['macro_vol']) & (data['enhanced_fractal_asym'] < 0)
    
    # Regime-specific alphas
    high_vol_alpha = data['vol_momentum_divergence'] * data['volume_acceleration'] * data['entropy_momentum_convergence'] * data['integrated_breakout_confidence']
    low_vol_alpha = data['volume_momentum_correlation'] * data['intraday_range_efficiency'] * data['fractal_entropy_alignment'] * data['multi_scale_gap_entropy']
    expanding_vol_alpha = data['opening_gap_momentum'] * data['gap_close_efficiency'] * data['amount_entropy_signal'] * data['multi_scale_entropy_acceleration']
    contracting_vol_alpha = data['volume_persistence'] * data['micro_meso_divergence'] * data['multi_fractal_volume_micro'] * data['fractal_momentum_intensity']
    
    # Final Alpha Synthesis with Regime Selection
    alpha = pd.Series(index=data.index, dtype=float)
    
    # Apply regime-specific alphas
    alpha[high_vol_condition] = high_vol_alpha[high_vol_condition]
    alpha[low_vol_condition] = low_vol_alpha[low_vol_condition]
    alpha[expanding_vol_condition] = expanding_vol_alpha[expanding_vol_condition]
    alpha[contracting_vol_condition] = contracting_vol_alpha[contracting_vol_condition]
    
    # Transition regime (average of all four)
    transition_condition = ~(high_vol_condition | low_vol_condition | expanding_vol_condition | contracting_vol_condition)
    alpha[transition_condition] = (high_vol_alpha + low_vol_alpha + expanding_vol_alpha + contracting_vol_alpha)[transition_condition] / 4
    
    # Entropy-Momentum Scaling
    entropy_momentum_scaling = alpha * (1 + abs(data['micro_meso_divergence']) * data['vol_entropy_div_signal'])
    
    # Final Multi-Scale Volatility-Entropy Momentum Alpha
    final_alpha = entropy_momentum_scaling * data['volume_persistence'] * data['fractal_momentum_intensity']
    
    return final_alpha
