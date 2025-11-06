import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Quantum Gap Momentum Calculation
    df = df.copy()
    
    # Overnight gap momentum
    df['gap_momentum'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # 3-day quantum gap persistence (alignment across days)
    df['gap_persistence'] = df['gap_momentum'].rolling(window=3).apply(
        lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) == 3 and not np.isnan(x).any() else 0
    )
    
    # Volatility-Quantum Elasticity Scaling
    df['quantum_elasticity'] = np.abs(df['close'] - df['open']) / (df['high'] - df['low'])
    df['quantum_elasticity'] = df['quantum_elasticity'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # 5-day volatility for scaling
    df['volatility'] = df['close'].pct_change().rolling(window=5).std()
    df['scaled_gap_momentum'] = df['gap_momentum'] * df['quantum_elasticity'] * df['volatility'].replace(0, 1)
    
    # Efficiency-Quantum Validation
    df['daily_efficiency'] = (df['close'] - df['open']) / (df['high'] - df['low'])
    df['daily_efficiency'] = df['daily_efficiency'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Quantum coherence (price consistency)
    df['quantum_coherence'] = 1 - (np.abs(df['close'] - df['open']) / df['close'])
    df['quantum_efficiency'] = df['daily_efficiency'] * df['quantum_coherence']
    
    # 5-day quantum efficiency trend
    df['efficiency_trend'] = df['quantum_efficiency'].rolling(window=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else 0
    )
    
    # Volume-Quantum Pressure Integration
    df['quantum_pressure'] = (df['volume'] / df['amount']) * df['daily_efficiency']
    df['quantum_pressure'] = df['quantum_pressure'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Weight momentum by quantum pressure confirmation
    df['pressure_weighted_momentum'] = df['scaled_gap_momentum'] * df['quantum_pressure']
    
    # Cross-Timeframe Quantum Convergence
    # 2-day vs 5-day momentum alignment
    df['momentum_2d'] = df['close'].pct_change(periods=2)
    df['momentum_5d'] = df['close'].pct_change(periods=5)
    
    df['momentum_alignment'] = np.sign(df['momentum_2d']) * np.sign(df['momentum_5d'])
    df['momentum_alignment'] = df['momentum_alignment'].fillna(0)
    
    # Final alpha factor construction
    df['quantum_alpha'] = (
        df['pressure_weighted_momentum'] * 
        df['efficiency_trend'] * 
        df['momentum_alignment'] * 
        df['gap_persistence']
    )
    
    # Clean and return
    alpha = df['quantum_alpha'].replace([np.inf, -np.inf], 0).fillna(0)
    return alpha
