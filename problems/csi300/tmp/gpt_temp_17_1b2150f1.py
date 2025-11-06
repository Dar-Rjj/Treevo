import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Multi-Scale Liquidity-Entropy Framework
    # Liquidity-Entropy Divergence
    # Micro Liquidity-Entropy
    micro_liq_entropy = ((data['close'] - data['close'].shift(1)) * data['volume'] / 
                        (data['high'] - data['low']) * 
                        (data['high'] - data['low']) / 
                        (np.abs(data['close'] - data['open']) + 1e-8))
    
    # Meso Liquidity-Entropy
    def meso_liq_entropy_func(data, t):
        close_diff = data['close'] - data['close'].shift(3)
        vol_sum = (data['high'].rolling(window=3).apply(lambda x: (x - data.loc[x.index, 'low']).sum(), raw=False) * 
                  data['volume'].rolling(window=3).sum())
        high_low_range = data['high'].rolling(window=3).max() - data['low'].rolling(window=3).min()
        return (close_diff * data['volume'] / (vol_sum + 1e-8) * 
                high_low_range / (np.abs(data['close'] - data['close'].shift(3)) + 1e-8))
    
    meso_liq_entropy = meso_liq_entropy_func(data, 3)
    
    # Macro Liquidity-Entropy
    def macro_liq_entropy_func(data, t):
        close_diff = data['close'] - data['close'].shift(5)
        vol_sum = (data['high'].rolling(window=5).apply(lambda x: (x - data.loc[x.index, 'low']).sum(), raw=False) * 
                  data['volume'].rolling(window=5).sum())
        high_low_range = data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min()
        return (close_diff * data['volume'] / (vol_sum + 1e-8) * 
                high_low_range / (np.abs(data['close'] - data['close'].shift(5)) + 1e-8))
    
    macro_liq_entropy = macro_liq_entropy_func(data, 5)
    
    # Liquidity-Entropy Divergence
    liq_entropy_divergence = (micro_liq_entropy - meso_liq_entropy) + (meso_liq_entropy - macro_liq_entropy)
    
    # Pressure-Entropy Dynamics
    # Opening Pressure-Entropy
    opening_pressure_entropy = ((data['open'] - data['low']) * data['volume'] / 
                               (np.abs(data['open'] - data['close'].shift(1)) + 1e-8) * 
                               (data['high'] - data['low']) / 
                               (np.abs(data['close'] - data['open']) + 1e-8))
    
    # Closing Pressure-Entropy
    closing_pressure_entropy = ((data['high'] - data['close']) * data['volume'] / 
                               (np.abs(data['close'] - data['open']) + 1e-8) * 
                               (data['high'] - data['low']) / 
                               (np.abs(data['close'] - data['open']) + 1e-8))
    
    # Pressure-Entropy Asymmetry
    pressure_entropy_asymmetry = opening_pressure_entropy - closing_pressure_entropy
    
    # Entropy Turnover Ratio
    short_turnover = (data['volume'] * data['close']).rolling(window=5).mean()
    long_turnover = (data['volume'] * data['close']).rolling(window=15).mean()
    entropy_turnover_ratio = (short_turnover / (long_turnover + 1e-8)) - 1
    
    # Quantum Liquidity Synthesis
    # Liquidity-Entropy Coupling
    liq_entropy_coupling = liq_entropy_divergence * pressure_entropy_asymmetry
    
    # Volume-Entropy Correlation
    volume_entropy_correlation = (data['volume'] * pressure_entropy_asymmetry / 
                                 (data['high'] - data['low'] + 1e-8))
    
    # Quantum Liquidity Alpha
    quantum_liquidity_alpha = liq_entropy_coupling * volume_entropy_correlation
    
    # Fractal Entropy Confirmation
    # Volume-Entropy Cluster Dynamics
    # Gap Entropy Fractal
    gap_volume = data['volume'] * (data['high'] - data['low'])
    max_gap_long = gap_volume.rolling(window=7).max()
    min_gap_long = gap_volume.rolling(window=7).min()
    max_gap_short = gap_volume.rolling(window=2).max()
    min_gap_short = gap_volume.rolling(window=2).min()
    gap_entropy_fractal = np.log((max_gap_long - min_gap_long + 1e-8) / 
                                (max_gap_short - min_gap_short + 1e-8))
    
    # Entropy Turnover Momentum
    max_turnover = (data['volume'] * data['close']).rolling(window=4).max()
    entropy_turnover_momentum = ((data['volume'] * data['close'] / (max_turnover.shift(1) + 1e-8)) * 
                                (data['high'] - data['low']) / 
                                (np.abs(data['close'] - data['open']) + 1e-8))
    
    # Entropy Cluster Duration
    turnover = data['volume'] * data['close']
    median_turnover = turnover.rolling(window=7).median()
    high_turnover = turnover > (2.5 * median_turnover)
    
    def count_consecutive(series):
        count = 0
        result = []
        for val in series:
            if val:
                count += 1
            else:
                count = 0
            result.append(count)
        return pd.Series(result, index=series.index)
    
    entropy_cluster_duration = count_consecutive(high_turnover)
    
    # Volume-Entropy Cluster Dynamics
    volume_entropy_cluster = (gap_entropy_fractal * entropy_turnover_momentum * 
                             entropy_cluster_duration)
    
    # Entropy Flow Asymmetry
    # Entropy Momentum
    entropy_momentum = ((data['volume'] / data['volume'].shift(1)) - 
                       (data['volume'].shift(1) / data['volume'].shift(2))) * \
                      ((data['high'] - data['low']) / 
                       (data['high'].shift(1) - data['low'].shift(1) + 1e-8))
    
    # Amount Entropy Efficiency
    amount_entropy_efficiency = ((data['amount'] / data['volume']) / 
                                (data['amount'].shift(1) / data['volume'].shift(1) + 1e-8)) * \
                               ((data['high'] - data['low']) / 
                                (np.abs(data['close'] - data['open']) + 1e-8))
    
    # Entropy Upside Ratio
    up_days = data['close'] > data['open']
    volume_up = data['volume'].where(up_days, 0)
    avg_volume_up = volume_up.rolling(window=10).mean()
    avg_volume_total = data['volume'].rolling(window=10).mean()
    entropy_upside_ratio = avg_volume_up / (avg_volume_total + 1e-8)
    
    # Entropy Flow Asymmetry
    entropy_flow_asymmetry = entropy_momentum * amount_entropy_efficiency * entropy_upside_ratio
    
    # Fractal Entropy Integration
    # Volume-Entropy Cluster
    volume_entropy_cluster_final = volume_entropy_cluster * entropy_flow_asymmetry
    
    # Entropy Flow Efficiency
    entropy_flow_efficiency = ((data['close'] - data['open']) * data['volume'] / 
                              (data['amount'] + 1e-8) * 
                              (data['high'] - data['low']) / 
                              (np.abs(data['close'] - data['open']) + 1e-8))
    
    # Fractal Entropy Alpha
    fractal_entropy_alpha = volume_entropy_cluster_final * entropy_flow_efficiency
    
    # Quantum Entropy-Volatility Mechanics
    # Entropy-Volatility Compression
    # Gap Entropy Compression
    gap_entropy_compression = (np.abs(data['close'] - data['open']).rolling(window=5).sum() / 
                              np.abs(data['close'] - data['open']).rolling(window=10).sum() - 1)
    
    # Entropy Range Expansion
    entropy_range_expansion = (data['high'].rolling(window=20).max() / 
                              data['low'].rolling(window=20).min() - 1)
    
    # Entropy-Volatility State
    entropy_volatility_state = gap_entropy_compression * entropy_range_expansion
    
    # Quantum Entropy Momentum
    # Intraday Quantum Momentum
    intraday_quantum_momentum = ((data['close'] - data['open']) * data['volume'] / 
                                (data['high'] - data['low'] + 1e-8) * 
                                (data['high'] - data['low']) / 
                                (np.abs(data['close'] - data['open']) + 1e-8))
    
    # Short-term Quantum Momentum
    short_term_quantum_momentum = (np.sign(data['close'] - data['close'].shift(3)) * 
                                  (data['high'].rolling(window=3).max() - data['low'].rolling(window=3).min()) / 
                                  (np.abs(data['close'] - data['close'].shift(3)) + 1e-8))
    
    # Medium-term Quantum Momentum
    medium_term_quantum_momentum = (np.sign(data['close'] - data['close'].shift(5)) * 
                                   (data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min()) / 
                                   (np.abs(data['close'] - data['close'].shift(5)) + 1e-8))
    
    # Quantum Momentum Convergence
    quantum_momentum_convergence = (np.sign(intraday_quantum_momentum) * 
                                   short_term_quantum_momentum * medium_term_quantum_momentum)
    
    # Quantum Momentum State
    quantum_momentum_state = quantum_momentum_convergence * intraday_quantum_momentum
    
    # Quantum Volatility Alpha
    quantum_volatility_alpha = entropy_volatility_state * quantum_momentum_state
    
    # Information-Entropy Cascade Detection
    # Multi-Dimensional Information Flow
    # Price Information Flow
    price_info_flow = (np.abs(data['close'] / data['close'].shift(1) - 1) - 
                      np.abs(data['close'].shift(1) / data['close'].shift(2) - 1))
    
    # Volume Information Flow
    volume_info_flow = (data['volume'] / data['volume'].shift(1) - 
                       data['volume'].shift(1) / data['volume'].shift(2))
    
    # Volatility Information Flow
    volatility_info_flow = ((data['high'] - data['low']) / 
                           (data['high'].shift(1) - data['low'].shift(1) + 1e-8) - 1)
    
    # Entropy Information Flow
    entropy_info_flow = ((data['high'] - data['low']) / (np.abs(data['close'] - data['open']) + 1e-8) - 
                        (data['high'].shift(1) - data['low'].shift(1)) / 
                        (np.abs(data['close'].shift(1) - data['open'].shift(1)) + 1e-8))
    
    # Quantum Entropy Cascade
    positive_quantum_cascade = ((price_info_flow > 0) & (volume_info_flow > 0) & 
                               (entropy_info_flow > 0))
    negative_quantum_cascade = ((price_info_flow < 0) & (volume_info_flow < 0) & 
                               (entropy_info_flow < 0))
    
    quantum_cascade_strength = np.where(positive_quantum_cascade, 
                                       price_info_flow * volume_info_flow * entropy_info_flow,
                                       np.where(negative_quantum_cascade, 
                                               price_info_flow * volume_info_flow * entropy_info_flow, 0))
    
    # Quantum Cascade Integration
    # Quantum Cascade Pressure
    quantum_cascade_pressure = quantum_cascade_strength * quantum_liquidity_alpha
    
    # Entropy Flow Quantum
    entropy_flow_quantum = ((data['close'] - data['open']) * data['volume'] / 
                           (data['amount'] + 1e-8) * 
                           (data['high'] - data['low']) / 
                           (np.abs(data['close'] - data['open']) + 1e-8))
    
    # Quantum Cascade Alpha
    quantum_cascade_alpha = quantum_cascade_pressure * entropy_flow_quantum
    
    # Quantum Breakout Confirmation
    # Multi-Dimensional Breakout Detection
    # Quantum Price Breakout
    quantum_price_breakout = ((data['close'] > data['high'].shift(1)) & 
                             ((data['close'] - data['open']) / (data['high'] - data['low'] + 0.001) > 0.6))
    
    # Volume Flow Breakout
    volume_flow_breakout = ((data['volume'] > 1.8 * data['volume'].shift(1)) & 
                           (data['amount'] > 1.7 * data['amount'].shift(1)))
    
    # Entropy Breakout
    entropy_breakout = (((data['high'] - data['low']) / (np.abs(data['close'] - data['open']) + 1e-8)) > 
                       (1.8 * (data['high'].shift(1) - data['low'].shift(1)) / 
                        (np.abs(data['close'].shift(1) - data['open'].shift(1)) + 1e-8)))
    
    # Quantum Breakout Assessment
    # Multi-Breakout Score
    multi_breakout_score = (quantum_price_breakout.astype(int) + 
                           volume_flow_breakout.astype(int) + 
                           entropy_breakout.astype(int))
    
    # Quantum Breakout Intensity
    quantum_breakout_intensity = (multi_breakout_score * 
                                np.abs(data['close'] - data['close'].shift(1)) / 
                                (data['high'] - data['low'] + 1e-8))
    
    # Quantum Breakout Persistence
    quantum_breakout_persistence = (quantum_breakout_intensity * 
                                  (data['volume'] / data['volume'].shift(1)) * 
                                  (data['amount'] / data['amount'].shift(1)))
    
    # Signal Enhancement
    # Quantum Breakout Amplifier
    quantum_breakout_amplifier = 1 + (quantum_breakout_persistence * multi_breakout_score)
    
    # Quantum Consolidation Dampener
    quantum_consolidation_dampener = 1 - ((data['volume'] < 0.7 * data['volume'].shift(1)) & 
                                         ((data['high'] - data['low']) < 0.8 * (data['high'].shift(1) - data['low'].shift(1))))
    
    # Enhanced Quantum Alpha
    enhanced_quantum_alpha = quantum_cascade_alpha * quantum_breakout_amplifier * quantum_consolidation_dampener
    
    # Final Quantum Liquidity-Entropy Synthesis
    # Base Quantum Signal
    base_quantum_signal = quantum_liquidity_alpha * fractal_entropy_alpha
    
    # Quantum Momentum Enhancement
    quantum_momentum_enhanced = base_quantum_signal * quantum_volatility_alpha
    
    # Quantum Cascade Integration
    quantum_cascade_integrated = quantum_momentum_enhanced * quantum_cascade_alpha
    
    # Quantum Breakout Finalization
    final_alpha = quantum_cascade_integrated * enhanced_quantum_alpha
    
    return final_alpha
