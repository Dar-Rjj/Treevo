import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate rolling windows
    for i in range(len(df)):
        if i < 10:  # Need at least 10 days of history
            result.iloc[i] = 0
            continue
            
        current = df.iloc[i]
        # Historical data (current and past only)
        hist_5 = df.iloc[max(0, i-5):i+1]  # t-5 to t
        hist_5_excl_current = df.iloc[max(0, i-5):i]  # t-5 to t-1
        hist_10 = df.iloc[max(0, i-10):i]  # t-10 to t-1
        hist_20 = df.iloc[max(0, i-20):i]  # t-20 to t-1
        
        # Fractal Gap Momentum Components
        # Overnight Fractal Gap
        if i > 0:
            gap_magnitude = (current['open'] - df.iloc[i-1]['close']) / df.iloc[i-1]['close']
            fractal_context = (current['open'] - hist_5_excl_current['low'].min()) / (hist_5_excl_current['high'].max() - hist_5_excl_current['low'].min())
            adjusted_gap = gap_magnitude * fractal_context
        else:
            adjusted_gap = 0
        
        # Intraday Fractal Recovery
        if i > 0:
            recovery_strength = (current['close'] - current['open']) / abs(current['open'] - df.iloc[i-1]['close'])
            fractal_efficiency = (current['close'] - hist_5['low'].min()) / (hist_5['high'].max() - hist_5['low'].min())
            enhanced_recovery = recovery_strength * fractal_efficiency
        else:
            enhanced_recovery = 0
        
        # Gap Persistence Analysis
        fractal_persistence = np.sign(adjusted_gap) * (current['close'] - current['open']) / (current['high'] - current['low'])
        volume_persistence = (current['volume'] / hist_5_excl_current['volume'].mean()) * np.sign(current['close'] - current['open'])
        combined_persistence = fractal_persistence * volume_persistence
        
        # Asymmetric Microstructure Rejection
        # Fractal Rejection Signals
        upper_shadow_rejection = (current['high'] - current['close']) / (current['high'] - current['low'])
        lower_shadow_acceptance = (current['close'] - current['low']) / (current['high'] - current['low'])
        net_fractal_rejection = lower_shadow_acceptance - upper_shadow_rejection
        
        # Rejection Momentum
        if i >= 3:
            short_term_momentum = sum([(df.iloc[j]['close'] - df.iloc[j]['low']) / (df.iloc[j]['high'] - df.iloc[j]['low']) - 
                                     (df.iloc[j]['high'] - df.iloc[j]['close']) / (df.iloc[j]['high'] - df.iloc[j]['low']) 
                                     for j in range(max(0, i-3), i)])
        else:
            short_term_momentum = 0
            
        if i >= 8:
            medium_term_momentum = sum([(df.iloc[j]['close'] - df.iloc[j]['low']) / (df.iloc[j]['high'] - df.iloc[j]['low']) - 
                                      (df.iloc[j]['high'] - df.iloc[j]['close']) / (df.iloc[j]['high'] - df.iloc[j]['low']) 
                                      for j in range(max(0, i-8), i)])
        else:
            medium_term_momentum = 0
            
        momentum_convergence = short_term_momentum * medium_term_momentum
        
        # Volume-Enhanced Rejection
        rejection_efficiency = net_fractal_rejection * (current['volume'] / (current['high'] - current['low']))
        volume_confirmation = (current['volume'] / hist_10['volume'].max()) * np.sign(net_fractal_rejection)
        enhanced_rejection = rejection_efficiency * volume_confirmation
        
        # Asymmetric Liquidity Dynamics
        # Directional Spread Analysis
        if current['close'] > current['open']:
            bullish_spread_pressure = (current['high'] - current['open']) / current['close']
            bearish_spread_pressure = 1  # Avoid division by zero
        else:
            bullish_spread_pressure = 1
            bearish_spread_pressure = (current['open'] - current['low']) / current['close']
        spread_asymmetry = bullish_spread_pressure / bearish_spread_pressure
        
        # Volume-Liquidity Efficiency
        if current['close'] > current['open']:
            bullish_efficiency = current['volume'] / (current['high'] - current['low'])
            bearish_efficiency = 1
        else:
            bullish_efficiency = 1
            bearish_efficiency = current['volume'] / (current['high'] - current['low'])
        efficiency_asymmetry = bullish_efficiency / bearish_efficiency
        
        # Fractal Order Flow
        upper_fractal_flow = abs(current['close'] - hist_5['high'].max()) / ((current['high'] - current['low'])/2) if current['close'] < hist_5['high'].max() else 0
        lower_fractal_flow = abs(current['close'] - hist_5['low'].min()) / ((current['high'] - current['low'])/2) if current['close'] > hist_5['low'].min() else 0
        net_fractal_flow = lower_fractal_flow - upper_fractal_flow
        
        # Regime-Based Integration
        # Gap-Dominated Regime
        gap_momentum_component = adjusted_gap * enhanced_recovery
        microstructure_confirmation = enhanced_rejection * combined_persistence
        liquidity_adjustment = efficiency_asymmetry * spread_asymmetry
        gap_dominated_alpha = 0.5 * gap_momentum_component + 0.3 * microstructure_confirmation + 0.2 * liquidity_adjustment
        
        # Microstructure-Dominated Regime
        fractal_rejection_component = momentum_convergence * enhanced_rejection
        order_flow_component = net_fractal_flow * volume_confirmation
        microstructure_dominated_alpha = 0.4 * fractal_rejection_component + 0.4 * order_flow_component + 0.2 * liquidity_adjustment
        
        # Breakout Regime
        volume_breakout = (current['volume'] / hist_10['volume'].max()) * np.sign(current['close'] - current['open'])
        fractal_breakout = (current['close'] - hist_5['low'].min()) / (hist_5['high'].max() - hist_5['low'].min())
        combined_breakout = volume_breakout * fractal_breakout
        breakout_alpha = combined_breakout * efficiency_asymmetry
        
        # Adaptive Alpha Synthesis
        # Regime Selection
        if abs(adjusted_gap) > 0.02:
            selected_alpha = gap_dominated_alpha
        elif current['volume'] > 2 * hist_10['volume'].mean():
            selected_alpha = breakout_alpha
        else:
            selected_alpha = microstructure_dominated_alpha
        
        # Volume Confirmation
        volume_multiplier = current['volume'] / hist_10['volume'].mean()
        volume_adjusted_alpha = selected_alpha * volume_multiplier
        
        # Final Enhancement
        fractal_momentum = np.sign(net_fractal_rejection) * momentum_convergence
        liquidity_filter = 1 / (1 + abs(spread_asymmetry - 1))
        final_alpha = volume_adjusted_alpha * fractal_momentum * liquidity_filter
        
        result.iloc[i] = final_alpha
    
    return result
