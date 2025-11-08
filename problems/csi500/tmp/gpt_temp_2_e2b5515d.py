import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining multiple technical indicators:
    1. Intraday Volatility-Adjusted Price Momentum
    2. Volume-Synchronized Price Probability
    3. Bid-Ask Spread Implied Momentum
    4. Liquidity-Adjusted Trend Strength
    5. Opening Gap Mean Reversion Probability
    """
    
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # 1. Intraday Volatility-Adjusted Price Momentum
    # Calculate short-term (5-day) and long-term (20-day) moving averages
    short_ma = data['close'].rolling(window=5).mean()
    long_ma = data['close'].rolling(window=20).mean()
    price_momentum = (short_ma - long_ma) / long_ma
    
    # Calculate volatility regime using 20-day rolling standard deviation
    returns = data['close'].pct_change()
    volatility = returns.rolling(window=20).std()
    volatility_regime = volatility.rolling(window=20).mean()
    
    # Scale momentum based on volatility regime
    volatility_adjustment = volatility_regime / volatility_regime.median()
    adjusted_momentum = price_momentum / np.sqrt(volatility_adjustment)
    
    # 2. Volume-Synchronized Price Probability
    # Calculate volume-weighted price percentiles
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    volume_weighted_price = (typical_price * data['volume']).rolling(window=10).sum() / data['volume'].rolling(window=10).sum()
    
    # Calculate current price's percentile rank
    price_percentile = data['close'].rolling(window=20).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) > 0 else 0.5
    )
    
    # Volume-synchronized signal (reversal at extremes)
    volume_signal = -np.abs(price_percentile - 0.5) * 2 + 1
    
    # 3. Bid-Ask Spread Implied Momentum
    # Estimate effective spread using Corwin-Schultz method
    high_low_ratio = np.log(data['high'] / data['low'])
    high_low_squared = high_low_ratio ** 2
    
    beta = high_low_squared.rolling(window=2).sum()
    gamma = (data['high'].rolling(window=2).max() / data['low'].rolling(window=2).min()).apply(np.log) ** 2
    
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * np.sqrt(2)) - np.sqrt(gamma / (3 - 2 * np.sqrt(2)))
    spread_estimate = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    spread_estimate = spread_estimate.replace([np.inf, -np.inf], np.nan).fillna(method='ffill')
    
    # Price momentum (5-day rate of change)
    price_roc = data['close'].pct_change(5)
    
    # Spread-adjusted momentum
    spread_signal = price_roc / (spread_estimate.rolling(window=10).mean() + 1e-8)
    
    # 4. Liquidity-Adjusted Trend Strength
    # Calculate ADX-like trend strength
    high_low_range = data['high'] - data['low']
    high_close_prev = np.abs(data['high'] - data['close'].shift(1))
    low_close_prev = np.abs(data['low'] - data['close'].shift(1))
    
    tr = pd.concat([high_low_range, high_close_prev, low_close_prev], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()
    
    # Simplified trend strength (price change relative to ATR)
    trend_strength = (data['close'] - data['close'].shift(5)) / atr
    
    # Calculate Amihud illiquidity ratio
    amihud_ratio = (np.abs(returns) / (data['volume'] * data['close'])).rolling(window=10).mean()
    
    # Liquidity adjustment (invert Amihud ratio for liquidity measure)
    liquidity_adjustment = 1 / (amihud_ratio + 1e-8)
    liquidity_adjusted_trend = trend_strength * liquidity_adjustment / liquidity_adjustment.median()
    
    # 5. Opening Gap Mean Reversion Probability
    # Calculate opening gap
    opening_gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Historical gap reversion statistics
    gap_reversion = opening_gap.rolling(window=20).apply(
        lambda x: -np.sign(x.iloc[-1]) if len(x) > 1 else 0
    )
    
    # Volume confirmation for gap signals
    volume_ratio = data['volume'] / data['volume'].rolling(window=20).mean()
    gap_signal = gap_reversion * np.abs(opening_gap) * (2 - volume_ratio)
    
    # Combine all signals with equal weights
    factor = (
        0.2 * adjusted_momentum.fillna(0) +
        0.2 * volume_signal.fillna(0) +
        0.2 * spread_signal.fillna(0) +
        0.2 * liquidity_adjusted_trend.fillna(0) +
        0.2 * gap_signal.fillna(0)
    )
    
    return factor
