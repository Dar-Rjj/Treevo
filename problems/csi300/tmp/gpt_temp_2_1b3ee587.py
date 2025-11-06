import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining multiple technical approaches:
    - Volatility-adjusted momentum
    - Volume-price divergence
    - Intraday strength persistence
    - Order flow imbalance
    - Multi-timeframe momentum convergence
    """
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Ensure we have enough data for calculations
    if len(df) < 50:
        return result
    
    # 1. Dynamic Volatility-Adjusted Momentum
    # Calculate 20-day price momentum
    momentum_20 = df['close'].pct_change(periods=20)
    
    # Compute 20-day volatility from High-Low ranges (annualized)
    daily_range = (df['high'] - df['low']) / df['close'].shift(1)
    volatility_20 = daily_range.rolling(window=20).std() * np.sqrt(252)
    
    # Avoid division by zero
    volatility_20 = volatility_20.replace(0, np.nan)
    vol_adjusted_momentum = momentum_20 / volatility_20
    
    # 2. Volume-Price Trend Divergence
    # Calculate 10-day rolling price trend (linear regression slope)
    price_trend = df['close'].rolling(window=10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else np.nan,
        raw=False
    )
    
    # Calculate 10-day rolling volume trend
    volume_trend = df['volume'].rolling(window=10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else np.nan,
        raw=False
    )
    
    # Compute divergence (z-score of difference)
    trend_divergence = (price_trend - volume_trend).rolling(window=20).apply(
        lambda x: (x[-1] - x.mean()) / x.std() if len(x) == 20 and x.std() > 0 else 0,
        raw=False
    )
    
    # 3. Intraday Strength Persistence
    # Calculate intraday strength ratio
    intraday_strength = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)
    
    # Measure persistence via 5-day rolling autocorrelation
    strength_persistence = intraday_strength.rolling(window=10).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if len(x) == 10 else 0,
        raw=False
    )
    
    # Combine current strength with persistence
    strength_signal = intraday_strength * strength_persistence
    
    # 4. Amount-Based Order Flow Imbalance
    # Classify trades using price movement and Amount data
    price_change = df['close'].pct_change()
    # Simple classification: positive return suggests buyer-initiated
    buyer_amount = df['amount'].where(price_change > 0, 0)
    seller_amount = df['amount'].where(price_change <= 0, 0)
    
    # Calculate 5-day net buyer-seller imbalance ratio
    buyer_rolling = buyer_amount.rolling(window=5).sum()
    seller_rolling = seller_amount.rolling(window=5).sum()
    
    order_imbalance = (buyer_rolling - seller_rolling) / (buyer_rolling + seller_rolling).replace(0, np.nan)
    
    # 5. Multi-timeframe Momentum Convergence
    # Calculate momentum across different windows
    momentum_5 = df['close'].pct_change(periods=5)
    momentum_10 = df['close'].pct_change(periods=10)
    momentum_20 = df['close'].pct_change(periods=20)
    
    # Compute convergence signal (all moving in same direction)
    momentum_convergence = np.sign(momentum_5) * np.sign(momentum_10) * np.sign(momentum_20)
    # Weight by average momentum strength
    avg_momentum = (momentum_5.abs() + momentum_10.abs() + momentum_20.abs()) / 3
    convergence_signal = momentum_convergence * avg_momentum
    
    # Combine all signals with equal weights
    signals = pd.DataFrame({
        'vol_momentum': vol_adjusted_momentum,
        'trend_div': trend_divergence,
        'strength': strength_signal,
        'order_flow': order_imbalance,
        'convergence': convergence_signal
    })
    
    # Normalize each signal using z-score (20-day rolling)
    for col in signals.columns:
        signals[col] = signals[col].rolling(window=20).apply(
            lambda x: (x[-1] - x.mean()) / x.std() if len(x) == 20 and x.std() > 0 else 0,
            raw=False
        )
    
    # Final factor: equal-weighted combination
    result = signals.mean(axis=1)
    
    return result
