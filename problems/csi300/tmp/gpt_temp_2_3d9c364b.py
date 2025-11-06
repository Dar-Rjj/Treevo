import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Quantum-inspired market microstructure alpha factor combining:
    - Price entanglement patterns through correlation tunneling
    - Volume-wave interference
    - Liquidity field dynamics
    - Market microstructure entropy
    """
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # Required columns
    cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
    for col in cols:
        if col not in df.columns:
            df[col] = 0.0
    
    # Calculate basic price features
    df['returns'] = df['close'].pct_change()
    df['price_range'] = (df['high'] - df['low']) / df['close']
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    
    # Quantum Price Entanglement - Multi-timeframe correlation tunneling
    # Use rolling correlations between different price components
    window_short = 5
    window_medium = 13
    window_long = 21
    
    # Short-term price entanglement (quantum correlation tunneling)
    df['entanglement_short'] = (
        df['close'].rolling(window=window_short).corr(df['typical_price']) * 
        df['returns'].rolling(window=window_short).std()
    )
    
    # Medium-term entanglement decay
    df['entanglement_medium'] = (
        df['close'].rolling(window=window_medium).corr(df['typical_price']) * 
        np.exp(-1/window_medium)  # Quantum decay factor
    )
    
    # Market Microstructure Wave-Particle Duality
    # Volume-wave interference patterns
    df['volume_wave'] = (
        df['volume'].rolling(window=window_short).mean() / 
        (df['volume'].rolling(window=window_medium).std() + 1e-8)
    )
    
    # Price particle momentum uncertainty (Heisenberg-inspired)
    df['momentum_uncertainty'] = (
        df['returns'].rolling(window=window_short).std() * 
        df['price_range'].rolling(window=window_short).std()
    )
    
    # Quantum Liquidity Field Dynamics
    # Liquidity potential well depth (amount/volume ratio dynamics)
    df['liquidity_potential'] = (
        (df['amount'] / (df['volume'] + 1e-8)).rolling(window=window_medium).mean() / 
        (df['amount'] / (df['volume'] + 1e-8)).rolling(window=window_long).std()
    )
    
    # Volume quantum state transitions (regime changes)
    df['volume_quantum_state'] = (
        df['volume'].pct_change().rolling(window=window_short).apply(
            lambda x: np.sum(np.abs(np.diff(x > 0)))  # Count state transitions
        )
    )
    
    # Market Microstructure Entropy Systems
    # Information entropy in price movements
    def price_entropy(series):
        returns = series.pct_change().dropna()
        if len(returns) < 2:
            return 0
        # Calculate probability distribution of return signs
        pos_prob = (returns > 0).mean()
        neg_prob = (returns < 0).mean()
        zero_prob = (returns == 0).mean()
        
        # Shannon entropy
        probs = [p for p in [pos_prob, neg_prob, zero_prob] if p > 0]
        return -sum(p * np.log(p) for p in probs)
    
    df['price_entropy'] = df['close'].rolling(window=window_medium).apply(price_entropy, raw=False)
    
    # Thermodynamic efficiency (price movement per unit volume)
    df['thermo_efficiency'] = (
        df['returns'].abs().rolling(window=window_short).mean() / 
        (df['volume'].pct_change().abs().rolling(window=window_short).mean() + 1e-8)
    )
    
    # Combine quantum-inspired components
    alpha = (
        # Quantum entanglement patterns (normalized)
        (df['entanglement_short'] - df['entanglement_short'].rolling(window=window_long).mean()) /
        (df['entanglement_short'].rolling(window=window_long).std() + 1e-8) +
        
        # Wave-particle duality effects
        df['volume_wave'] * (1 / (df['momentum_uncertainty'] + 1e-8)) +
        
        # Liquidity field dynamics
        df['liquidity_potential'].fillna(0) * df['volume_quantum_state'].fillna(0) +
        
        # Entropy reversal patterns (negative correlation with inefficiency)
        -df['price_entropy'].fillna(0) * df['thermo_efficiency'].fillna(0)
    )
    
    # Final normalization and cleaning
    alpha = alpha.fillna(0)
    alpha = (alpha - alpha.rolling(window=63, min_periods=1).mean()) / (alpha.rolling(window=63, min_periods=1).std() + 1e-8)
    
    return alpha
