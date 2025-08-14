import pandas as pd
from arch import arch_model

def heuristics_v2(df):
    # Decompose the close price using Hilbert transform (trend and cycle)
    from scipy.signal import hilbert
    analytic_signal = hilbert(df['close'])
    amplitude_envelope = pd.Series(abs(analytic_signal), index=df.index)
    
    # Fit GARCH(1,1) model for conditional variance as a measure of volatility clustering
    garch_model = arch_model(df['close'].pct_change().dropna(), vol='Garch', p=1, q=1)
    garch_result = garch_model.fit(disp='off')
    cond_volatility = garch_result.conditional_volatility
    
    # Momentum factor - Rate of Change over 60 days
    roc_60 = df['close'].pct_change(60)
    
    # Composite heuristic
    heuristics_matrix = (amplitude_envelope + cond_volatility + roc_60) / 3
    return heuristics_matrix
