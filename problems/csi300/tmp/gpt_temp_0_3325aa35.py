import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Quantum Momentum Topology Factor
    Combines momentum superposition, topological persistence, and volatility decoherence
    to identify coherent multi-scale momentum regimes.
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    factor = pd.Series(index=df.index, dtype=float)
    
    # Quantum Momentum State Construction
    # Calculate momentum superposition across multiple time horizons
    for i in range(len(df)):
        if i < 20:  # Need sufficient history
            factor.iloc[i] = 0
            continue
            
        current_data = df.iloc[:i+1]  # Only use current and past data
        
        # Multi-scale momentum calculation
        momentum_states = []
        
        # Short-term momentum (5-day)
        if i >= 5:
            short_momentum = (current_data['close'].iloc[i] / current_data['close'].iloc[i-5] - 1) * 100
            momentum_states.append(short_momentum)
        
        # Medium-term momentum (10-day)
        if i >= 10:
            medium_momentum = (current_data['close'].iloc[i] / current_data['close'].iloc[i-10] - 1) * 100
            momentum_states.append(medium_momentum)
        
        # Long-term momentum (20-day)
        long_momentum = (current_data['close'].iloc[i] / current_data['close'].iloc[i-20] - 1) * 100
        momentum_states.append(long_momentum)
        
        # Intraday momentum (current day)
        intraday_momentum = (current_data['close'].iloc[i] / current_data['open'].iloc[i] - 1) * 100
        momentum_states.append(intraday_momentum)
        
        # Calculate momentum superposition (weighted average)
        if len(momentum_states) >= 3:
            weights = [0.2, 0.3, 0.3, 0.2]  # Weight by time horizon importance
            momentum_superposition = sum(m * w for m, w in zip(momentum_states[:4], weights))
        else:
            momentum_superposition = 0
        
        # Topological Momentum Persistence
        # Build momentum manifolds from recent price velocity
        if i >= 10:
            recent_returns = []
            for j in range(1, 11):
                if i - j >= 0:
                    ret = (current_data['close'].iloc[i-j+1] / current_data['close'].iloc[i-j] - 1) * 100
                    recent_returns.append(ret)
            
            # Calculate momentum persistence (autocorrelation of returns)
            if len(recent_returns) >= 5:
                momentum_persistence = np.corrcoef(recent_returns[:5], recent_returns[5:10])[0,1] if len(recent_returns) >= 10 else 0
            else:
                momentum_persistence = 0
        else:
            momentum_persistence = 0
        
        # Volatility-Induced Decoherence Analysis
        # Use intraday volatility as measurement apparatus
        if i >= 5:
            # Calculate recent volatility (5-day rolling)
            recent_volatility = current_data['close'].iloc[i-4:i+1].pct_change().std() * np.sqrt(252) * 100
            if pd.isna(recent_volatility):
                recent_volatility = 0
            
            # Volatility clustering (GARCH-like effect)
            volatility_clustering = 0
            if i >= 10:
                vol_series = []
                for j in range(10):
                    if i - j - 1 >= 0:
                        vol = abs(current_data['close'].iloc[i-j] / current_data['close'].iloc[i-j-1] - 1) * 100
                        vol_series.append(vol)
                if len(vol_series) >= 5:
                    volatility_clustering = np.corrcoef(vol_series[:5], vol_series[5:10])[0,1] if len(vol_series) >= 10 else 0
        else:
            recent_volatility = 0
            volatility_clustering = 0
        
        # Multi-Scale Momentum Synchronization
        # Calculate transfer entropy between momentum scales
        momentum_sync = 0
        if i >= 15:
            # Calculate correlation between short and medium-term momentum changes
            short_changes = []
            medium_changes = []
            
            for j in range(5):
                if i - j - 1 >= 5:
                    short_prev = (current_data['close'].iloc[i-j-1] / current_data['close'].iloc[i-j-6] - 1) * 100
                    short_curr = (current_data['close'].iloc[i-j] / current_data['close'].iloc[i-j-5] - 1) * 100
                    short_changes.append(short_curr - short_prev)
                
                if i - j - 1 >= 10:
                    medium_prev = (current_data['close'].iloc[i-j-1] / current_data['close'].iloc[i-j-11] - 1) * 100
                    medium_curr = (current_data['close'].iloc[i-j] / current_data['close'].iloc[i-j-10] - 1) * 100
                    medium_changes.append(medium_curr - medium_prev)
            
            if len(short_changes) >= 3 and len(medium_changes) >= 3:
                momentum_sync = np.corrcoef(short_changes[:3], medium_changes[:3])[0,1] if len(short_changes) >= 3 and len(medium_changes) >= 3 else 0
        
        # Critical Momentum Compression
        # Calculate information density in momentum states
        momentum_compression = 0
        if i >= 10:
            # Measure how concentrated momentum is across time scales
            momentum_variation = np.std(momentum_states) if len(momentum_states) > 0 else 0
            avg_momentum = np.mean(momentum_states) if len(momentum_states) > 0 else 0
            if avg_momentum != 0:
                momentum_compression = abs(avg_momentum) / (momentum_variation + 1e-6)
        
        # Quantum-Topological Signal Generation
        # Combine all components with appropriate weights
        coherence_score = (
            0.4 * momentum_superposition +          # Momentum superposition
            0.3 * momentum_persistence +            # Topological persistence
            0.2 * momentum_sync -                   # Multi-scale synchronization
            0.1 * recent_volatility -               # Volatility decoherence penalty
            0.1 * volatility_clustering +           # Volatility clustering adjustment
            0.1 * momentum_compression              # Momentum compression
        )
        
        factor.iloc[i] = coherence_score
    
    # Normalize the factor
    if len(factor) > 0:
        factor = (factor - factor.mean()) / (factor.std() + 1e-6)
    
    return factor
