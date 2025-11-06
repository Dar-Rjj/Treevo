import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Stress-Efficiency Regime Classification
    # Volatility Stress Dynamics
    data['fractal_vol_stress'] = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1))
    data['volume_induced_stress'] = (data['volume'] * (data['high'] - data['low'])) / (data['volume'].shift(1) * (data['high'].shift(1) - data['low'].shift(1)))
    
    # Stress Persistence
    def calc_stress_persistence(series):
        if len(series) < 3:
            return np.nan
        signs = np.sign(series)
        persistence = sum(signs.iloc[i] == signs.iloc[i-1] for i in range(1, len(signs))) / (len(signs) - 1)
        return persistence
    
    data['stress_persistence'] = data['fractal_vol_stress'].rolling(window=3, min_periods=3).apply(calc_stress_persistence, raw=False)
    
    # Efficiency Regime State
    data['realized_efficiency'] = abs(data['close'] - data['open']) / (data['high'] - data['low'])
    data['volume_efficiency'] = data['volume'] / (data['volume'] + data['volume'].shift(1) + data['volume'].shift(2))
    data['efficiency_momentum'] = data['realized_efficiency'] / data['realized_efficiency'].shift(1) - 1
    
    # Microstructure Stress Patterns
    # Asymmetric Rejection Dynamics
    data['upside_stress'] = data['high'] - np.maximum(data['open'], data['close'])
    data['downside_stress'] = np.minimum(data['open'], data['close']) - data['low']
    data['net_stress_asymmetry'] = data['upside_stress'] - data['downside_stress']
    
    # Order Flow Pressure
    data['buying_stress'] = ((data['close'] - data['low']) / (data['high'] - data['low'])) * data['volume']
    data['selling_stress'] = ((data['high'] - data['close']) / (data['high'] - data['low'])) * data['volume']
    data['net_flow_stress'] = data['buying_stress'] - data['selling_stress']
    
    # Trade Impact Stress
    data['effective_spread'] = 2 * abs(data['close'] - (data['high'] + data['low'])/2) / ((data['high'] + data['low'])/2)
    data['trade_size_stress'] = (data['amount'] / data['volume']) / (data['amount'].shift(1) / data['volume'].shift(1)) - 1
    data['spread_absorption'] = data['volume'] * data['effective_spread'] / data['effective_spread'].shift(5)
    
    # Multi-Timeframe Stress-Efficiency Signals
    # Short-Term Stress Dynamics
    data['flow_stress_acceleration'] = ((data['close'] - data['close'].shift(1)) / data['close'].shift(1)) * (data['volume'] / data['volume'].shift(1))
    data['efficiency_stress_divergence'] = np.sign(data['realized_efficiency'] - data['realized_efficiency'].shift(1)) * np.sign(data['volume_induced_stress'])
    data['rejection_velocity_alignment'] = np.sign(data['net_stress_asymmetry']) * np.sign(data['volume'] / data['volume'].shift(1) - 1)
    
    # Medium-Term Accumulation Patterns
    data['smart_money_stress'] = ((data['close'] - data['low'].rolling(window=5).min()) / 
                                 (data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min())) * data['volume']
    data['volume_weighted_stress'] = data['volume_induced_stress'] * data['volume_efficiency']
    data['stress_breakout'] = ((data['high'] / data['high'].shift(1) - 1) - 
                              (data['low'] / data['low'].shift(1) - 1)) * data['volume_induced_stress']
    
    # Long-Term Distribution Signals
    data['institutional_stress'] = (data['amount'] / data['volume']) / (
        (data['amount'].shift(1) / data['volume'].shift(1) + 
         data['amount'].shift(2) / data['volume'].shift(2) + 
         data['amount'].shift(3) / data['volume'].shift(3) + 
         data['amount'].shift(4) / data['volume'].shift(4) + 
         data['amount'].shift(5) / data['volume'].shift(5)) / 5)
    data['large_trade_efficiency_stress'] = data['institutional_stress'] * data['realized_efficiency']
    data['volatility_compression'] = ((data['high'] - data['low']) / (data['high'].shift(5) - data['low'].shift(5))) * (data['volume'] / data['volume'].shift(5))
    
    # Stress-Efficiency Validation
    # Microstructure Confirmation
    data['flow_direction_consistency'] = np.sign(data['close'] - data['close'].shift(1)) * np.sign(data['volume'] - data['volume'].shift(1))
    
    def calc_stress_momentum_alignment(close_series):
        if len(close_series) < 3:
            return np.nan
        returns = close_series.pct_change()
        signs = np.sign(returns)
        alignment = sum(signs.iloc[i] == signs.iloc[i-1] for i in range(1, len(signs))) / (len(signs) - 1)
        return alignment
    
    data['stress_momentum_alignment'] = data['close'].rolling(window=3, min_periods=3).apply(calc_stress_momentum_alignment, raw=False)
    
    def calc_efficiency_persistence(efficiency_series):
        if len(efficiency_series) < 3:
            return np.nan
        changes = efficiency_series.diff()
        signs = np.sign(changes)
        persistence = sum(signs.iloc[i] == signs.iloc[i-1] for i in range(2, len(signs))) / (len(signs) - 2)
        return persistence
    
    data['efficiency_persistence'] = data['realized_efficiency'].rolling(window=3, min_periods=3).apply(calc_efficiency_persistence, raw=False)
    
    # Cross-Regime Validation
    data['stress_efficiency_divergence'] = (data['close'] / data['close'].shift(3) - 1) - (data['realized_efficiency'] / data['realized_efficiency'].shift(3) - 1)
    data['volume_stress_confirmation'] = (data['volume_efficiency'] / data['volume_efficiency'].shift(1) - 1) - (data['realized_efficiency'] / data['realized_efficiency'].shift(1) - 1)
    data['spread_stress_confirmation'] = -data['trade_size_stress'] * (data['close'] / data['close'].shift(1) - 1)
    
    # Adaptive Stress-Efficiency Alpha Synthesis
    # Core Stress-Efficiency Components
    data['short_term_core'] = data['net_flow_stress'] * data['flow_stress_acceleration'] * data['spread_stress_confirmation']
    data['medium_term_core'] = data['smart_money_stress'] * data['stress_breakout'] * data['volume_weighted_stress']
    data['long_term_core'] = data['large_trade_efficiency_stress'] * data['institutional_stress'] * data['volatility_compression']
    
    # Stress-Adaptive Weighting
    data['high_stress_weight'] = data['fractal_vol_stress'] / (1 + abs(data['net_stress_asymmetry']))
    data['low_stress_weight'] = data['volume_efficiency'] / (1 + abs(data['trade_size_stress']))
    data['transition_weight'] = data['spread_absorption'] / (1 + abs(data['efficiency_momentum']))
    
    # Final Alpha Construction
    data['primary_alpha'] = data['short_term_core'] * data['high_stress_weight'] * data['stress_persistence']
    data['secondary_alpha'] = data['medium_term_core'] * data['transition_weight'] * data['efficiency_stress_divergence']
    data['tertiary_alpha'] = data['long_term_core'] * data['low_stress_weight'] * data['stress_momentum_alignment']
    
    # Final Signal
    data['final_signal'] = (data['primary_alpha'] * data['rejection_velocity_alignment'] + 
                           data['secondary_alpha'] * data['flow_direction_consistency'] + 
                           data['tertiary_alpha'] * data['efficiency_persistence'])
    
    return data['final_signal']
