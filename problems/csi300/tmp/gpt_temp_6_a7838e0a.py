import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Multi-timeframe momentum with volatility normalization and signal combination
    # Factor captures stocks with consistent momentum across different time horizons,
    # supported by volume dynamics and adjusted for volatility regimes
    
    # Short-term momentum (5-day return)
    momentum_short = df['close'] / df['close'].shift(5) - 1
    
    # Medium-term momentum (20-day return)
    momentum_medium = df['close'] / df['close'].shift(20) - 1
    
    # Long-term momentum (60-day return)
    momentum_long = df['close'] / df['close'].shift(60) - 1
    
    # Volume momentum across different timeframes
    volume_momentum_short = df['volume'] / df['volume'].shift(5) - 1
    volume_momentum_medium = df['volume'] / df['volume'].shift(20) - 1
    
    # Volatility calculation using multiple methods
    # Daily price range volatility
    daily_range_vol = (df['high'] - df['low']) / df['close'].shift(1)
    
    # Close-to-close volatility (5-day rolling std)
    close_vol = df['close'].pct_change().rolling(window=5).std()
    
    # Combined volatility measure
    combined_volatility = daily_range_vol * close_vol
    
    # Volatility normalization using rolling percentiles (20-day window)
    vol_normalized = combined_volatility.rolling(window=20).apply(
        lambda x: (x.iloc[-1] - x.quantile(0.25)) / (x.quantile(0.75) - x.quantile(0.25) + 1e-7)
    )
    
    # Price efficiency signal (close relative to high-low range)
    price_efficiency = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-7)
    
    # Amount-based liquidity signal (dollar volume trend)
    amount_trend = df['amount'] / df['amount'].shift(5) - 1
    
    # Combine multiple signals with weights
    # Momentum convergence signal (all timeframes moving together)
    momentum_convergence = momentum_short * momentum_medium * momentum_long
    
    # Volume confirmation signal
    volume_confirmation = volume_momentum_short * volume_momentum_medium
    
    # Price efficiency persistence (5-day average)
    efficiency_persistence = price_efficiency.rolling(window=5).mean()
    
    # Final factor combining all signals with volatility adjustment
    factor = (
        momentum_convergence * 
        volume_confirmation * 
        efficiency_persistence * 
        amount_trend / 
        (vol_normalized + 1e-7)
    )
    
    return factor
