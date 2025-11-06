import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate a composite alpha factor using multiple market microstructure signals
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    alpha_factor = pd.Series(index=data.index, dtype=float)
    
    # 1. Price Momentum with Volatility Adjustment
    # Compute rolling price momentum (20-day returns)
    momentum_20d = data['close'].pct_change(20)
    
    # Calculate volatility regime using daily range percentiles
    daily_range = (data['high'] - data['low']) / data['close']
    vol_regime = daily_range.rolling(30).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    
    # Adjust momentum based on volatility regime (higher vol = more conservative)
    vol_adjusted_momentum = momentum_20d * (1 - vol_regime)
    
    # 2. Intraday Pattern Persistence
    # Calculate intraday price ratios
    high_close_ratio = data['high'] / data['close'] - 1
    low_close_ratio = 1 - data['low'] / data['close']
    
    # Detect consecutive pattern persistence (3-day same direction)
    high_persistence = ((high_close_ratio > 0) & (high_close_ratio.shift(1) > 0) & (high_close_ratio.shift(2) > 0)).astype(int)
    low_persistence = ((low_close_ratio > 0) & (low_close_ratio.shift(1) > 0) & (low_close_ratio.shift(2) > 0)).astype(int)
    
    # Generate persistence signals (positive for high persistence, negative for low persistence)
    pattern_signal = high_persistence - low_persistence
    
    # 3. Liquidity-Weighted Mean Reversion
    # Compute price deviation from 10-day rolling mean
    price_mean_10d = data['close'].rolling(10).mean()
    price_deviation = (data['close'] - price_mean_10d) / price_mean_10d
    
    # Assess liquidity via volume and amount (normalized)
    volume_ma = data['volume'].rolling(20).mean()
    amount_ma = data['amount'].rolling(20).mean()
    liquidity_score = (data['volume'] / volume_ma + data['amount'] / amount_ma) / 2
    
    # Generate liquidity-adjusted mean reversion score
    liquidity_reversion = -price_deviation * liquidity_score
    
    # 4. Volume-Confirmed Breakout
    # Identify key support/resistance levels (20-day high/low)
    resistance_20d = data['high'].rolling(20).max()
    support_20d = data['low'].rolling(20).min()
    
    # Analyze volume behavior at price levels
    volume_breakout = np.where(
        data['close'] > resistance_20d.shift(1), 
        data['volume'] / data['volume'].rolling(5).mean(),
        np.where(
            data['close'] < support_20d.shift(1),
            -data['volume'] / data['volume'].rolling(5).mean(),
            0
        )
    )
    
    # 5. Asymmetric Volatility Analysis
    # Calculate separate upside/downside volatility (10-day)
    returns = data['close'].pct_change()
    upside_vol = returns[returns > 0].rolling(10).std()
    downside_vol = (-returns[returns < 0]).rolling(10).std()
    
    # Compare volatility asymmetry ratios
    vol_asymmetry = (upside_vol - downside_vol) / (upside_vol + downside_vol + 1e-8)
    vol_asymmetry = vol_asymmetry.fillna(0)
    
    # Generate asymmetry-based momentum signals
    asymmetry_signal = vol_asymmetry * momentum_20d
    
    # 6. Efficiency-Adjusted Trend Following
    # Compute price efficiency ratio (absolute price change vs total movement)
    price_change = abs(data['close'] - data['close'].shift(5))
    total_movement = abs(data['close'] - data['open']).rolling(5).sum()
    efficiency_ratio = price_change / (total_movement + 1e-8)
    
    # Assess trend quality via efficiency (20-day trend)
    trend_20d = data['close'].rolling(20).apply(lambda x: (x[-1] - x[0]) / x[0], raw=True)
    
    # Generate efficiency-weighted trend signals
    efficiency_trend = trend_20d * efficiency_ratio
    
    # Combine all signals with equal weights
    combined_signals = (
        vol_adjusted_momentum.fillna(0) * 0.2 +
        pattern_signal.fillna(0) * 0.15 +
        liquidity_reversion.fillna(0) * 0.2 +
        volume_breakout * 0.15 +
        asymmetry_signal.fillna(0) * 0.15 +
        efficiency_trend.fillna(0) * 0.15
    )
    
    # Normalize the final factor using z-score (20-day rolling)
    alpha_factor = (combined_signals - combined_signals.rolling(20).mean()) / (combined_signals.rolling(20).std() + 1e-8)
    
    return alpha_factor
