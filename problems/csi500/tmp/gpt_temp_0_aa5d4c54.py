import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Volatility Regime Adaptive Momentum Factor
    
    # Regime Detection Component
    df['volatility_regime'] = (df['high'] - df['low']) / df['close']
    
    # Calculate regime persistence (count of increasing regime over 5 days)
    regime_increase = (df['volatility_regime'] > df['volatility_regime'].shift(1)).astype(int)
    df['regime_persistence'] = regime_increase.rolling(window=5, min_periods=1).sum()
    
    # Combine regime and persistence
    regime_component = df['volatility_regime'] * df['regime_persistence']
    
    # Adaptive Momentum Component
    df['raw_momentum'] = df['close'] / df['close'].shift(5) - 1
    
    # Volume confirmation
    volume_ratio = np.log(df['volume'] / df['volume'].shift(5))
    volume_ratio = volume_ratio.replace([np.inf, -np.inf], 0).fillna(0)
    df['volume_confirmation'] = np.sign(df['raw_momentum']) * volume_ratio
    
    # Combine momentum and volume confirmation
    momentum_component = df['raw_momentum'] * df['volume_confirmation']
    
    # Price Efficiency Component
    df['gap_efficiency'] = (df['open'] - df['close'].shift(1)) / (df['high'] - df['low'])
    df['gap_efficiency'] = df['gap_efficiency'].replace([np.inf, -np.inf], 0).fillna(0)
    
    df['close_efficiency'] = (df['close'] - df['open']) / (df['high'] - df['low'])
    df['close_efficiency'] = df['close_efficiency'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Combine efficiency components
    efficiency_component = np.abs(df['gap_efficiency']) * df['close_efficiency']
    
    # Final Factor Construction
    # Regime weighted momentum
    weighted_momentum = regime_component * momentum_component
    
    # Efficiency filtered
    efficiency_filtered = weighted_momentum * efficiency_component
    
    # Dynamic smoothing based on regime persistence
    smoothing_window = (df['regime_persistence'] + 2).astype(int)
    
    # Apply EMA with dynamic window
    factor_values = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i == 0:
            factor_values.iloc[i] = efficiency_filtered.iloc[i]
        else:
            window = min(smoothing_window.iloc[i], i + 1)
            weights = np.exp(-np.arange(window) / window)
            weights = weights / weights.sum()
            factor_values.iloc[i] = np.sum(efficiency_filtered.iloc[i - window + 1:i + 1] * weights)
    
    return factor_values.fillna(0)
