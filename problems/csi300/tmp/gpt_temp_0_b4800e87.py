import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Neural Synchronization Market Cognition
    # Calculate resting potential deviation at market open
    df['resting_potential'] = df['open'] - df['close'].shift(1)
    
    # Calculate action potential threshold breach
    df['depolarization'] = (df['high'] - df['open']) / df['open']
    
    # Calculate hyperpolarization limits
    df['hyperpolarization'] = (df['open'] - df['low']) / df['open']
    
    # Calculate spike-timing-dependent plasticity
    df['stdp'] = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    
    # Calculate neurotransmitter concentration gradient
    df['neurotransmitter_gradient'] = df['amount'] / df['amount'].rolling(window=5, min_periods=1).mean()
    
    # Calculate neural firing rate from volume
    df['firing_rate'] = df['volume'] / df['volume'].rolling(window=10, min_periods=1).mean()
    
    # Calculate phase-locking value between price and volume
    price_returns = df['close'].pct_change()
    volume_returns = df['volume'].pct_change()
    df['phase_locking'] = price_returns.rolling(window=5, min_periods=1).corr(volume_returns)
    
    # Detect neural avalanches through volatility clustering
    returns = df['close'].pct_change()
    df['volatility_cluster'] = returns.rolling(window=5, min_periods=1).std() / returns.rolling(window=20, min_periods=1).std()
    
    # Calculate branching ratio from price cascades
    df['price_cascade'] = (df['high'] - df['low']).rolling(window=3, min_periods=1).std() / (df['high'] - df['low']).rolling(window=10, min_periods=1).std()
    
    # Fluid Dynamic Information Flow
    # Calculate Reynolds number equivalent
    df['reynolds_number'] = (df['close'] - df['open']).abs() / (df['high'] - df['low']).replace(0, np.nan)
    
    # Calculate turbulent kinetic energy
    df['turbulent_energy'] = ((df['high'] - df['low']) / df['open']).rolling(window=5, min_periods=1).std()
    
    # Calculate information cascade through volatility scales
    short_vol = returns.rolling(window=3, min_periods=1).std()
    long_vol = returns.rolling(window=10, min_periods=1).std()
    df['richardson_cascade'] = short_vol / long_vol
    
    # Ecological Niche Competition Dynamics
    # Calculate carrying capacity deviation
    df['carrying_capacity'] = (df['close'] - df['close'].rolling(window=20, min_periods=1).mean()) / df['close'].rolling(window=20, min_periods=1).std()
    
    # Calculate niche overlap from correlation with volume
    df['niche_overlap'] = df['close'].rolling(window=10, min_periods=1).corr(df['volume'])
    
    # Calculate population growth rate
    df['growth_rate'] = df['volume'].pct_change(periods=3)
    
    # Cellular Signal Transduction Pathways
    # Calculate signal amplification
    df['signal_amplification'] = (df['high'] - df['open']) / (df['open'] - df['low']).replace(0, np.nan)
    
    # Calculate signal-to-noise ratio
    df['signal_noise_ratio'] = (df['close'] - df['open']).abs() / (df['high'] - df['low']).replace(0, np.nan)
    
    # Calculate adaptation time constant
    range_ratio = (df['high'] - df['low']) / df['open']
    df['adaptation_constant'] = range_ratio.rolling(window=5, min_periods=1).mean() / range_ratio.rolling(window=20, min_periods=1).mean()
    
    # Crystallographic Symmetry Breaking
    # Calculate mean square displacement
    df['msd'] = ((df['close'] - df['open']) / df['open']).rolling(window=5, min_periods=1).std()
    
    # Calculate order parameter (trend strength)
    df['order_parameter'] = (df['close'] - df['close'].rolling(window=10, min_periods=1).mean()).abs() / df['close'].rolling(window=10, min_periods=1).std()
    
    # Calculate correlation length
    autocorr_1 = df['close'].pct_change().rolling(window=5, min_periods=1).apply(lambda x: x.autocorr(lag=1), raw=False)
    autocorr_2 = df['close'].pct_change().rolling(window=5, min_periods=1).apply(lambda x: x.autocorr(lag=2), raw=False)
    df['correlation_length'] = autocorr_1 / autocorr_2.replace(0, np.nan)
    
    # Combine factors with appropriate weights
    factor = (
        0.15 * df['phase_locking'].fillna(0) +
        0.12 * df['volatility_cluster'].fillna(0) +
        0.10 * df['richardson_cascade'].fillna(0) +
        0.13 * df['carrying_capacity'].fillna(0) +
        0.11 * df['signal_noise_ratio'].fillna(0) +
        0.09 * df['adaptation_constant'].fillna(0) +
        0.10 * df['order_parameter'].fillna(0) +
        0.08 * df['correlation_length'].fillna(0) +
        0.12 * df['stdp'].fillna(0)
    )
    
    return factor
