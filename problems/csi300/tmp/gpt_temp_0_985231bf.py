import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Volatility-Weighted Intraday Efficiency
    intraday_efficiency = (df['close'] - df['open']) / (df['high'] - df['low'])
    intraday_efficiency = intraday_efficiency.replace([np.inf, -np.inf], np.nan)
    
    short_term_efficiency = intraday_efficiency.rolling(window=3, min_periods=1).mean()
    medium_term_efficiency = intraday_efficiency.rolling(window=10, min_periods=1).mean()
    efficiency_divergence = short_term_efficiency - medium_term_efficiency
    
    # Volatility-Adjusted Acceleration
    returns_1d = df['close'].pct_change(1)
    returns_3d = df['close'].pct_change(3)
    returns_5d = df['close'].pct_change(5)
    
    price_acceleration = (returns_3d - returns_1d) - (returns_5d - returns_3d)
    volatility_20d = returns_1d.rolling(window=20, min_periods=1).std()
    volatility_adjusted_acceleration = price_acceleration / volatility_20d
    volatility_adjusted_acceleration = volatility_adjusted_acceleration.replace([np.inf, -np.inf], np.nan)
    
    # Volume-Amount Confirmation
    volume_5d_avg = df['volume'].rolling(window=5, min_periods=1).mean()
    amount_5d_avg = df['amount'].rolling(window=5, min_periods=1).mean()
    
    volume_efficiency = df['volume'] / volume_5d_avg
    amount_concentration = df['amount'] / amount_5d_avg
    
    # Regime Classification
    eff_div_quantile = efficiency_divergence.rolling(window=20, min_periods=1).apply(
        lambda x: pd.qcut(x, q=4, labels=False, duplicates='drop').iloc[-1] if len(x) >= 4 else 2, 
        raw=False
    )
    
    acc_quantile = volatility_adjusted_acceleration.rolling(window=20, min_periods=1).apply(
        lambda x: pd.qcut(x, q=4, labels=False, duplicates='drop').iloc[-1] if len(x) >= 4 else 2,
        raw=False
    )
    
    # Regime mapping
    regime = pd.Series(index=df.index, dtype=float)
    regime[(eff_div_quantile >= 2) & (acc_quantile >= 2)] = 1.0  # Momentum regime
    regime[(eff_div_quantile >= 2) & (acc_quantile < 2)] = 0.5   # Efficiency regime  
    regime[(eff_div_quantile < 2) & (acc_quantile >= 2)] = -0.5  # Volatility regime
    regime[(eff_div_quantile < 2) & (acc_quantile < 2)] = -1.0   # Noise regime
    
    # Final factor combining all components
    factor = (
        efficiency_divergence * 0.4 + 
        volatility_adjusted_acceleration * 0.3 + 
        volume_efficiency * 0.15 + 
        amount_concentration * 0.15
    ) * regime
    
    return factor
