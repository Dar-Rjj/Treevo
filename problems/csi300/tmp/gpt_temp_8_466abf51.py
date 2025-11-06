import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Scale Volatility Regime Classification
    # Volatility Spectrum Analysis
    data['micro_vol'] = (data['high'].rolling(window=3).max() - data['low'].rolling(window=3).min()) / data['close'].shift(2)
    data['meso_vol'] = (data['high'].rolling(window=8).max() - data['low'].rolling(window=8).min()) / data['close'].shift(7)
    data['macro_vol'] = (data['high'].rolling(window=21).max() - data['low'].rolling(window=21).min()) / data['close'].shift(20)
    
    # Volatility Regime Identification
    data['high_vol_regime'] = (data['micro_vol'] > data['meso_vol']) & (data['meso_vol'] > data['macro_vol'])
    data['low_vol_regime'] = (data['micro_vol'] < data['meso_vol']) & (data['meso_vol'] < data['macro_vol'])
    data['transition_vol_regime'] = (data['micro_vol'] > data['meso_vol']) & (data['meso_vol'] < data['macro_vol'])
    
    # Volatility Persistence Score
    vol_diff_sign = np.sign(data['micro_vol'] - data['meso_vol'])
    data['volatility_direction_consistency'] = vol_diff_sign.rolling(window=5).apply(
        lambda x: (x == x.iloc[-1]).sum() if len(x) == 5 else np.nan, raw=False
    )
    data['regime_stability'] = data['volatility_direction_consistency'] / 5
    
    # Price-Volume Entropy Signals
    # Entropy Components
    data['price_entropy'] = -np.log(np.abs((data['close'] - data['open']) / (data['high'] - data['low'])) + 0.001)
    data['volume_entropy'] = -np.log(data['volume'] / (data['high'] - data['low']) + 0.001)
    data['entropy_divergence'] = data['price_entropy'] - data['volume_entropy']
    
    # Multi-Timeframe Entropy Dynamics
    data['short_term_entropy_change'] = data['entropy_divergence'] - data['entropy_divergence'].shift(3)
    data['medium_term_entropy_trend'] = data['entropy_divergence'] - data['entropy_divergence'].shift(8)
    data['entropy_acceleration'] = data['short_term_entropy_change'] - data['medium_term_entropy_trend']
    
    # Entropy-Persistence Composite
    data['entropy_signal'] = data['entropy_acceleration'] * data['regime_stability']
    entropy_signal_sign = np.sign(data['entropy_signal'])
    data['entropy_consistency'] = entropy_signal_sign.rolling(window=4).apply(
        lambda x: (x == x.iloc[-1]).sum() if len(x) == 4 else np.nan, raw=False
    )
    
    # Volume-Price Fractal Efficiency
    # Efficiency Components
    def calculate_efficiency(series, period):
        price_diff = series.diff()
        efficiency = price_diff / price_diff.abs().rolling(window=period).sum()
        return efficiency
    
    data['price_efficiency'] = calculate_efficiency(data['close'], 3)
    data['volume_efficiency'] = calculate_efficiency(data['volume'], 3)
    data['efficiency_ratio'] = data['price_efficiency'] / (data['volume_efficiency'] + 0.001)
    
    # Multi-Scale Efficiency Analysis
    data['short_term_efficiency'] = data['efficiency_ratio']
    data['medium_term_efficiency'] = calculate_efficiency(data['close'], 8)
    data['efficiency_convergence'] = data['short_term_efficiency'] - data['medium_term_efficiency']
    
    # Efficiency-Persistence Score
    data['efficiency_signal'] = data['efficiency_convergence'] * data['entropy_consistency']
    efficiency_signal_sign = np.sign(data['efficiency_signal'])
    data['efficiency_stability'] = efficiency_signal_sign.rolling(window=4).apply(
        lambda x: (x == x.iloc[-1]).sum() if len(x) == 4 else np.nan, raw=False
    )
    
    # Regime-Adaptive Alpha Synthesis
    # Base Entropy-Efficiency Core
    data['entropy_efficiency_composite'] = data['entropy_signal'] * data['efficiency_signal']
    data['core_persistence'] = data['entropy_consistency'] * data['efficiency_stability']
    
    # Volatility-Regime-Specific Enhancement
    data['final_alpha'] = np.nan
    
    # High Volatility Regime
    high_vol_mask = data['high_vol_regime']
    data.loc[high_vol_mask, 'final_alpha'] = (
        data.loc[high_vol_mask, 'entropy_efficiency_composite'] * 
        data.loc[high_vol_mask, 'micro_vol'] / 
        (data.loc[high_vol_mask, 'efficiency_stability'] + 0.001)
    )
    
    # Low Volatility Regime
    low_vol_mask = data['low_vol_regime']
    data.loc[low_vol_mask, 'final_alpha'] = (
        data.loc[low_vol_mask, 'entropy_efficiency_composite'] * 
        data.loc[low_vol_mask, 'efficiency_stability'] / 
        (data.loc[low_vol_mask, 'micro_vol'] + 0.001)
    )
    
    # Transition Volatility Regime
    transition_vol_mask = data['transition_vol_regime']
    data.loc[transition_vol_mask, 'final_alpha'] = (
        data.loc[transition_vol_mask, 'entropy_efficiency_composite'] * 
        data.loc[transition_vol_mask, 'entropy_consistency'] / 
        (np.abs(data.loc[transition_vol_mask, 'efficiency_convergence']) + 0.001)
    )
    
    # Fill any remaining NaN values with the composite signal
    data['final_alpha'] = data['final_alpha'].fillna(data['entropy_efficiency_composite'])
    
    return data['final_alpha']
