import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate returns
    returns = df['close'].pct_change()
    
    # Compute Asymmetric Price Momentum
    # Upside Momentum Strength
    pos_returns_5d = returns.rolling(window=5).apply(lambda x: x[x > 0].mean() if len(x[x > 0]) > 0 else 0)
    pos_returns_10d = returns.rolling(window=10).apply(lambda x: x[x > 0].mean() if len(x[x > 0]) > 0 else 0)
    upside_momentum_ratio = pos_returns_5d / pos_returns_10d
    
    # Downside Momentum Strength
    neg_returns_5d = returns.rolling(window=5).apply(lambda x: x[x < 0].mean() if len(x[x < 0]) > 0 else 0)
    neg_returns_10d = returns.rolling(window=10).apply(lambda x: x[x < 0].mean() if len(x[x < 0]) > 0 else 0)
    downside_momentum_ratio = neg_returns_5d / neg_returns_10d
    
    # Detect Fractal Market Regimes
    # Volatility Fractal Patterns
    short_term_range = (df['high'].rolling(window=3).max() - df['low'].rolling(window=3).min()) / df['close'].rolling(window=3).mean()
    medium_term_range = (df['high'].rolling(window=8).max() - df['low'].rolling(window=8).min()) / df['close'].rolling(window=8).mean()
    volatility_fractal_dimension = short_term_range / medium_term_range
    
    # Volume Fractal Structure
    short_term_volume_dispersion = (df['volume'].rolling(window=3).max() - df['volume'].rolling(window=3).min()) / df['volume'].rolling(window=3).mean()
    medium_term_volume_dispersion = (df['volume'].rolling(window=8).max() - df['volume'].rolling(window=8).min()) / df['volume'].rolling(window=8).mean()
    volume_fractal_ratio = short_term_volume_dispersion / medium_term_volume_dispersion
    
    # Generate Asymmetric Momentum Factor
    # Combine Upside and Downside Momentum
    net_momentum_asymmetry = upside_momentum_ratio - downside_momentum_ratio
    regime_adjusted_momentum = net_momentum_asymmetry * volatility_fractal_dimension
    
    # Incorporate Volume Structure Enhancement
    structure_enhanced_signal = regime_adjusted_momentum * volume_fractal_ratio
    
    # Apply Dynamic Response Filter
    def adaptive_rolling(x):
        fractal_dim = volatility_fractal_dimension.loc[x.index[-1]]
        window = 2 if fractal_dim > 1.2 else 4
        return x.rolling(window=window).mean().iloc[-1]
    
    final_factor = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 9:  # Ensure enough data for calculations
            window_data = structure_enhanced_signal.iloc[max(0, i-9):i+1]
            final_factor.iloc[i] = adaptive_rolling(window_data)
    
    return final_factor
