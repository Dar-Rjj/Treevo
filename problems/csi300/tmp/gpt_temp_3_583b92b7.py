import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate novel alpha factor combining momentum, volume confirmation, 
    volume-spike reversal, liquidity impact, intraday patterns, and regime-adaptive mean reversion.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Momentum with Volatility Scaling
    # Short-term momentum (5-day)
    mom_short = data['close'] / data['close'].shift(5) - 1
    
    # Medium-term momentum (20-day)
    mom_medium = data['close'] / data['close'].shift(20) - 1
    
    # Volatility adjustment using high-low range (20-day rolling)
    hl_range = (data['high'] - data['low']) / data['close']
    vol_20d = hl_range.rolling(window=20).std()
    
    # Volume trend slope (5-day linear regression)
    volume_trend = data['volume'].rolling(window=5).apply(
        lambda x: np.polyfit(range(5), x, 1)[0], raw=False
    )
    
    # Combined momentum factor
    momentum_factor = (mom_short * 0.6 + mom_medium * 0.4) / (vol_20d + 1e-8)
    momentum_factor = momentum_factor * np.tanh(volume_trend / (data['volume'].rolling(20).std() + 1e-8))
    
    # 2. Volume-Spike Reversal
    # Recent return (t-2 to t)
    recent_return = data['close'] / data['close'].shift(2) - 1
    
    # Historical average return (t-20 to t-3)
    hist_return = (data['close'].shift(3) / data['close'].shift(20) - 1).rolling(window=18).mean()
    
    # Volume z-score (20-day rolling)
    volume_zscore = (data['volume'] - data['volume'].rolling(window=20).mean()) / (data['volume'].rolling(window=20).std() + 1e-8)
    
    # Volume spike detection
    volume_spike = (volume_zscore > 2).astype(int)
    
    # Reversal signal
    extreme_return = np.abs(recent_return - hist_return) > 2 * recent_return.rolling(window=20).std()
    reversal_signal = -recent_return * extreme_return * volume_spike
    
    # 3. Liquidity Price Impact
    # Price change per unit volume
    price_impact = (data['close'] - data['open']) / (data['volume'] + 1e-8)
    
    # Turnover efficiency
    turnover_eff = data['amount'] / (data['volume'] + 1e-8)
    
    # High-low range as spread proxy
    spread_proxy = (data['high'] - data['low']) / data['close']
    
    # Volume concentration (intraday volume pattern)
    volume_concentration = data['volume'] / data['volume'].rolling(window=5).mean()
    
    # Liquidity factor
    liquidity_factor = -price_impact * turnover_eff / (spread_proxy + 1e-8) * volume_concentration
    
    # 4. Intraday Pattern Persistence
    # Close position in daily range
    range_position = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    
    # Range utilization ratio
    range_utilization = (data['high'] - data['low']) / data['close'].shift(1)
    
    # Consecutive strength days
    strength_days = (range_position > 0.6).rolling(window=3).sum()
    
    # Pattern persistence
    intraday_factor = range_position * range_utilization * strength_days
    
    # 5. Regime-Adaptive Mean Reversion
    # Volatility regime classification
    volatility_regime = hl_range.rolling(window=20).std() > hl_range.rolling(window=60).std()
    
    # Trend vs range-bound detection
    trend_strength = data['close'].rolling(window=20).apply(
        lambda x: np.polyfit(range(20), x, 1)[0], raw=False
    ) / (data['close'].rolling(window=20).std() + 1e-8)
    
    range_bound = np.abs(trend_strength) < 0.5
    
    # Price deviation from local mean (20-day MA)
    price_deviation = (data['close'] - data['close'].rolling(window=20).mean()) / (data['close'].rolling(window=20).std() + 1e-8)
    
    # Regime-adjusted reversion
    regime_factor = -price_deviation * (~volatility_regime) * range_bound
    
    # 6. Combined Alpha Factor
    # Normalize individual factors
    factors = pd.DataFrame({
        'momentum': momentum_factor,
        'reversal': reversal_signal,
        'liquidity': liquidity_factor,
        'intraday': intraday_factor,
        'regime': regime_factor
    })
    
    # Z-score normalization
    normalized_factors = (factors - factors.rolling(window=60).mean()) / (factors.rolling(window=60).std() + 1e-8)
    
    # Weighted combination
    weights = [0.25, 0.20, 0.20, 0.15, 0.20]  # momentum, reversal, liquidity, intraday, regime
    alpha_factor = normalized_factors.dot(weights)
    
    # Final smoothing and scaling
    alpha_factor = alpha_factor.rolling(window=3).mean()
    alpha_factor = alpha_factor / (alpha_factor.rolling(window=60).std() + 1e-8)
    
    return alpha_factor
