import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

def heuristics_v2(df):
    """
    Novel alpha factor combining microstructure patterns, regime transitions, 
    and information geometry signals using only current and historical data.
    """
    result = pd.Series(index=df.index, dtype=float)
    
    # Initialize rolling windows
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    amount = df['amount']
    
    for i in range(20, len(df)):
        current_data = df.iloc[:i+1]  # Only use data up to current day
        
        # 1. Volatility spillover intensity (Cross-Market Information Flow)
        hl_range = (high.iloc[i-19:i+1] - low.iloc[:i+1].iloc[i-19:i+1]) / close.iloc[:i+1].iloc[i-19:i+1]
        vol_spillover = hl_range.std() * np.sqrt(252)  # Annualized volatility
        
        # 2. Volume-entanglement (Non-Linear Price Relationships)
        volume_returns = volume.iloc[i-9:i+1].pct_change().dropna()
        price_returns = close.iloc[i-9:i+1].pct_change().dropna()
        if len(volume_returns) > 1 and len(price_returns) > 1:
            try:
                vol_entanglement = abs(pearsonr(volume_returns, price_returns)[0])
            except:
                vol_entanglement = 0
        else:
            vol_entanglement = 0
        
        # 3. Information leakage through trading gaps (Temporal Microstructure Patterns)
        volume_gap = volume.iloc[i] / volume.iloc[i-5:i].mean() if i >= 5 else 1.0
        price_gap = abs(close.iloc[i] - close.iloc[i-1]) / close.iloc[i-1]
        info_leakage = volume_gap * price_gap
        
        # 4. Critical slowing down before regime shifts (Market Regime Transitions)
        recent_returns = close.iloc[i-9:i+1].pct_change().dropna()
        if len(recent_returns) > 1:
            autocorr_lag1 = recent_returns.autocorr(lag=1)
            critical_slowing = 1 - autocorr_lag1 if not pd.isna(autocorr_lag1) else 0
        else:
            critical_slowing = 0
        
        # 5. Fisher information from price distributions (Information Geometry Signals)
        returns_20d = close.iloc[i-19:i+1].pct_change().dropna()
        if len(returns_20d) > 1:
            fisher_info = 1 / (returns_20d.var() + 1e-8)  # Inverse of variance
        else:
            fisher_info = 0
        
        # Combine components with appropriate weights
        factor_value = (
            0.25 * vol_spillover +
            0.20 * vol_entanglement +
            0.15 * info_leakage +
            0.25 * critical_slowing +
            0.15 * fisher_info
        )
        
        result.iloc[i] = factor_value
    
    # Fill initial values with 0
    result = result.fillna(0)
    
    return result
