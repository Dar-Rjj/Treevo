import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Quantum-Topological Momentum Synthesis factor combining multi-scale quantum state persistence,
    volume-induced decoherence analysis, momentum acceleration, and gap/breakout topological signatures.
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Multi-Scale Quantum State Persistence
    def quantum_coherence_metric(close, window):
        """Calculate quantum coherence across different time horizons"""
        # Wavefunction amplitude based on price momentum
        momentum = close.pct_change(window)
        # Quantum state stability (persistence of momentum direction)
        state_stability = momentum.rolling(window=5).std() / (momentum.rolling(window=5).mean().abs() + 1e-8)
        return 1 / (1 + state_stability)
    
    # Calculate multi-scale coherence
    coherence_3d = quantum_coherence_metric(df['close'], 3)
    coherence_5d = quantum_coherence_metric(df['close'], 5)
    coherence_10d = quantum_coherence_metric(df['close'], 10)
    
    # Topological protection assessment
    def topological_protection(high, low, close, window=10):
        """Measure topological order through price path persistence"""
        # Persistent homology proxy: price range stability
        daily_range = (high - low) / close
        range_stability = daily_range.rolling(window=window).std() / (daily_range.rolling(window=window).mean() + 1e-8)
        
        # Trend persistence (Betti number proxy)
        trend_strength = close.rolling(window=5).apply(lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 else 0)
        
        return (1 / (1 + range_stability)) * np.abs(trend_strength)
    
    topo_protection = topological_protection(df['high'], df['low'], df['close'])
    
    # Volume-Induced Decoherence & Flow Efficiency
    def volume_decoherence_analysis(volume, amount, close, window=10):
        """Analyze quantum decoherence through volume patterns"""
        # Volume spike induced state collapse
        volume_zscore = (volume - volume.rolling(window=window).mean()) / (volume.rolling(window=window).std() + 1e-8)
        decoherence_prob = np.exp(-np.abs(volume_zscore) / 2)
        
        # Money flow divergence
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * volume
        money_flow_ratio = money_flow / (money_flow.rolling(window=5).mean() + 1e-8)
        
        # Flow efficiency (volume-price correlation)
        volume_price_corr = volume.rolling(window=window).corr(close)
        
        return decoherence_prob * money_flow_ratio * (1 + volume_price_corr)
    
    flow_efficiency = volume_decoherence_analysis(df['volume'], df['amount'], df['close'])
    
    # Momentum Acceleration in Quantum-Topological Space
    def quantum_momentum_entanglement(close, high, low, window=10):
        """Calculate momentum entanglement across multiple timeframes"""
        # Multi-period momentum differences
        mom_3d = close.pct_change(3)
        mom_5d = close.pct_change(5)
        mom_10d = close.pct_change(10)
        
        # Momentum entanglement (correlation between different timeframe momentums)
        mom_corr_3_5 = mom_3d.rolling(window=5).corr(mom_5d)
        mom_corr_5_10 = mom_5d.rolling(window=5).corr(mom_10d)
        
        # Momentum tunneling through resistance (ability to maintain momentum through volatility)
        volatility = (high - low) / close
        momentum_persistence = (mom_3d.abs() + mom_5d.abs() + mom_10d.abs()) / (3 * volatility + 1e-8)
        
        return (mom_corr_3_5 + mom_corr_5_10) * momentum_persistence
    
    momentum_entanglement = quantum_momentum_entanglement(df['close'], df['high'], df['low'])
    
    # Gap & Breakout Quantum-Topological Signatures
    def gap_breakout_signatures(open_price, close, high, low, window=10):
        """Analyze gap reactions and breakout topological patterns"""
        # Overnight gap superposition
        overnight_gap = (open_price - close.shift(1)) / close.shift(1)
        
        # Gap filling/continuation quantum interference
        gap_reaction = np.where(
            overnight_gap > 0,
            (high - open_price) / (overnight_gap * close.shift(1) + 1e-8),  # Up gap reaction
            (open_price - low) / (-overnight_gap * close.shift(1) + 1e-8)   # Down gap reaction
        )
        
        # Breakout topological validation (persistent homology changes)
        breakout_strength = (high.rolling(window=5).max() - low.rolling(window=5).min()) / close
        breakout_persistence = breakout_strength / (breakout_strength.rolling(window=10).std() + 1e-8)
        
        return gap_reaction * breakout_persistence
    
    gap_signatures = gap_breakout_signatures(df['open'], df['close'], df['high'], df['low'])
    
    # Signal Generation Framework
    def quantum_topological_coherence_scoring():
        """Combine all components into final factor"""
        # Multi-scale coherence weighted by topological protection
        coherence_score = (coherence_3d * 0.3 + coherence_5d * 0.4 + coherence_10d * 0.3) * topo_protection
        
        # Momentum entanglement strength
        momentum_score = momentum_entanglement * np.tanh(np.abs(momentum_entanglement))
        
        # Gap reaction patterns
        gap_score = gap_signatures * np.sign(gap_signatures)
        
        # Volume flow efficiency adjustment
        flow_adjustment = np.log1p(np.abs(flow_efficiency)) * np.sign(flow_efficiency)
        
        # Final quantum-topological factor
        quantum_factor = (
            coherence_score * 0.25 +
            momentum_score * 0.30 +
            gap_score * 0.25 +
            flow_adjustment * 0.20
        )
        
        return quantum_factor
    
    # Calculate the final factor
    result = quantum_topological_coherence_scoring()
    
    # Remove any potential lookahead bias by ensuring only past data is used
    result = result.shift(1)  # Use yesterday's calculated value for today's prediction
    
    return result
