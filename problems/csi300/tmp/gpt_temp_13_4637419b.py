import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Quantum Field Fluctuations
    # Calculate price vacuum expectation values using rolling mean
    price_vacuum = (df['open'] + df['high'] + df['low'] + df['close']).rolling(window=20).mean()
    
    # Identify price quantum excitation levels using price volatility
    price_excitation = (df['high'] - df['low']).rolling(window=10).std() / df['close'].rolling(window=10).mean()
    
    # Compute price field correlation functions using autocorrelation
    price_correlation = df['close'].rolling(window=15).apply(lambda x: x.autocorr(lag=1), raw=False)
    
    # Volume Particle Creation Dynamics
    # Analyze volume quantum tunneling probabilities using volume momentum
    volume_tunneling = (df['volume'] - df['volume'].rolling(window=10).mean()) / df['volume'].rolling(window=10).std()
    
    # Calculate volume field interaction strengths using volume-price relationship
    volume_interaction = (df['volume'] * df['close']).rolling(window=10).corr(df['close'])
    
    # Detect volume spontaneous symmetry breaking using volume asymmetry
    volume_symmetry_break = (df['volume'] - df['volume'].shift(1)).rolling(window=5).apply(
        lambda x: np.sum(x > 0) - np.sum(x < 0), raw=False
    )
    
    # Quantum Excitation Signal
    # Combine field fluctuations and particle dynamics
    field_fluctuation_component = (price_excitation * price_correlation).fillna(0)
    particle_dynamics_component = (volume_tunneling * volume_interaction).fillna(0)
    
    # Calculate quantum excitation momentum
    quantum_momentum = (field_fluctuation_component.rolling(window=5).mean() + 
                       particle_dynamics_component.rolling(window=5).mean())
    
    # Generate field-based predictive signals with normalization
    quantum_signal = (quantum_momentum - quantum_momentum.rolling(window=20).mean()) / quantum_momentum.rolling(window=20).std()
    
    # Output quantum excitation alpha factor
    alpha_factor = quantum_signal.fillna(0)
    
    return alpha_factor
