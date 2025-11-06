import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Asset Microstructure Contagion Alpha Factor
    Captures delayed price reactions and cross-market inefficiencies
    using only current and historical data
    """
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate rolling statistics for cross-market comparison
    # Use 20-day window for medium-term patterns
    window = 20
    
    # Price momentum and volatility measures
    returns = df['close'].pct_change()
    volatility = returns.rolling(window=5).std()
    
    # Volume-based liquidity measures
    volume_ma = df['volume'].rolling(window=window).mean()
    volume_ratio = df['volume'] / volume_ma
    
    # Price range efficiency (high-low relative to close)
    price_range = (df['high'] - df['low']) / df['close']
    range_efficiency = price_range.rolling(window=window).mean()
    
    # Amount-based order flow intensity
    amount_ma = df['amount'].rolling(window=window).mean()
    flow_intensity = df['amount'] / amount_ma
    
    # Cross-market inefficiency detection
    # 1. Delayed reaction to volatility shocks
    vol_shock = (volatility - volatility.rolling(window=window).mean()) / volatility.rolling(window=window).std()
    vol_reaction_delay = vol_shock.rolling(window=5).mean()
    
    # 2. Price momentum persistence across regimes
    momentum_persistence = returns.rolling(window=5).apply(
        lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 and not np.isnan(x).any() else 0
    )
    
    # 3. Liquidity migration signal
    # When volume increases but price movement is muted, suggests cross-market flow
    liquidity_migration = (volume_ratio - price_range.rolling(window=5).mean()) / price_range.rolling(window=5).std()
    
    # 4. Order flow momentum
    flow_momentum = flow_intensity.rolling(window=5).mean()
    
    # 5. Contagion risk detection
    # High volatility with declining volume suggests cross-market stress
    contagion_risk = (volatility - volatility.rolling(window=window).mean()) * (1 - volume_ratio)
    
    # Combine signals with appropriate weights
    # Focus on cross-market inefficiency patterns
    alpha_signal = (
        0.3 * vol_reaction_delay +           # Delayed volatility response
        0.25 * momentum_persistence +        # Momentum persistence
        0.2 * liquidity_migration +          # Liquidity migration
        0.15 * flow_momentum +               # Order flow momentum
        0.1 * contagion_risk                 # Contagion risk
    )
    
    # Normalize the final signal
    result = (alpha_signal - alpha_signal.rolling(window=window).mean()) / alpha_signal.rolling(window=window).std()
    
    # Handle NaN values
    result = result.fillna(0)
    
    return result
