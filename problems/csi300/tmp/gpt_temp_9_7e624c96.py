import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Momentum Breakout factor
    """
    # Calculate returns
    returns = df['close'].pct_change()
    
    # Volatility Regime Classification
    vol_20d = returns.rolling(window=20).std()
    vol_median_60d = vol_20d.rolling(window=60).median()
    volatility_regime = (vol_20d > vol_median_60d).astype(int)
    
    # Multi-Timeframe Breakout Detection
    high_20d = df['high'].rolling(window=20).max()
    low_20d = df['low'].rolling(window=20).min()
    high_50d = df['high'].rolling(window=50).max()
    low_50d = df['low'].rolling(window=50).min()
    
    # Breakout strength calculation
    upper_breakout_strength = (df['close'] - high_20d) / high_20d
    lower_breakout_strength = (df['close'] - low_20d) / low_20d
    breakout_strength = np.where(df['close'] > high_20d, upper_breakout_strength,
                                np.where(df['close'] < low_20d, lower_breakout_strength, 0))
    
    # Breakout persistence (3-day maintenance)
    breakout_persistence = pd.Series(breakout_strength, index=df.index).rolling(window=3).mean()
    
    # Range efficiency
    daily_range = df['high'] - df['low']
    closing_position = (df['close'] - df['low']) / (daily_range + 1e-8)
    range_efficiency = closing_position * (1 - abs(closing_position - 0.5))
    
    # Volume-Price Synergy
    volume_20d_avg = df['volume'].rolling(window=20).mean()
    volume_ratio = df['volume'] / (volume_20d_avg + 1e-8)
    
    # Volume surge detection
    volume_surge = (volume_ratio > 2).astype(int)
    price_action_surge = ((df['close'] - df['open']) / (daily_range + 1e-8)) * volume_surge
    
    # Intraday momentum structure (simplified - using open/high/low/close)
    morning_momentum = (df['high'] - df['open']) / (daily_range + 1e-8)
    afternoon_momentum = (df['close'] - df['high']) / (daily_range + 1e-8)
    
    # Regime-Adaptive Signal Construction
    # High volatility signal
    return_10d = df['close'].pct_change(10)
    high_vol_signal = (return_10d / (vol_20d + 1e-8)) * volume_ratio * breakout_strength
    
    # Low volatility signal
    momentum_persistence = returns.rolling(window=5).apply(lambda x: pd.Series(x).autocorr(), raw=False)
    low_vol_signal = return_10d * momentum_persistence * range_efficiency * volume_ratio
    
    # Volume contraction adaptation
    volume_contraction = (volume_ratio < 0.8).astype(int)
    intraday_reversal = morning_momentum * afternoon_momentum * volume_contraction
    
    # Combine regime signals
    regime_signal = (volatility_regime * high_vol_signal + 
                    (1 - volatility_regime) * low_vol_signal)
    
    # Apply volume contraction adaptation
    regime_signal = regime_signal * (1 - 0.5 * volume_contraction) + intraday_reversal * volume_contraction
    
    # Liquidity Assessment
    price_trend_slope = df['close'].rolling(window=10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else 0, raw=False
    )
    liquidity_score = volume_ratio * (1 + abs(price_trend_slope))
    
    # Signal adjustment based on liquidity
    liquidity_adjustment = np.where(liquidity_score > 1.2, 1.2, 
                                   np.where(liquidity_score < 0.8, 0.8, 1.0))
    
    # Multi-Dimensional Confirmation
    # Timeframe alignment
    short_term_alignment = morning_momentum * breakout_persistence
    medium_term_alignment = breakout_strength * breakout_persistence
    
    # Volume-price coherence
    volume_price_coherence = price_action_surge * regime_signal
    
    # Final signal integration
    final_signal = (regime_signal * liquidity_adjustment * 
                   (0.6 + 0.2 * short_term_alignment + 0.2 * medium_term_alignment) *
                   (0.8 + 0.2 * volume_price_coherence))
    
    return pd.Series(final_signal, index=df.index, name='regime_adaptive_momentum_breakout')
