import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Generate alpha factor combining volatility-normalized price-volume divergence,
    regime-aware range efficiency, volume-scaled extreme reversal, and amount-based flow persistence.
    """
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Volatility-Normalized Price-Volume Divergence
    # Multi-timeframe momentum
    short_momentum = df['close'].pct_change(3)
    medium_momentum = df['close'].pct_change(10)
    momentum_ratio = short_momentum / (medium_momentum + 1e-8)
    
    # Volume persistence analysis
    volume_trend = df['volume'].rolling(window=5).apply(
        lambda x: stats.linregress(range(5), x)[0], raw=False
    )
    volume_increase_days = (df['volume'] > df['volume'].shift(1)).rolling(5).sum()
    volume_persistence = volume_trend * volume_increase_days
    
    # True range for volatility normalization
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    avg_true_range = true_range.rolling(5).mean()
    
    # Price-volume divergence
    price_volume_alignment = momentum_ratio * volume_persistence
    pv_divergence = price_volume_alignment / (avg_true_range + 1e-8)
    
    # Regime-Aware Range Efficiency
    range_utilization = abs(df['close'].pct_change()) / (true_range / df['close'].shift(1) + 1e-8)
    
    # Volatility regime detection
    volatility_regime = (true_range / df['close'].shift(1)).rolling(20).mean()
    high_vol_regime = volatility_regime > volatility_regime.quantile(0.7)
    low_vol_regime = volatility_regime < volatility_regime.quantile(0.3)
    
    # Regime-adaptive efficiency
    high_vol_efficiency = range_utilization * high_vol_regime.astype(int)
    low_vol_efficiency = range_utilization * low_vol_regime.astype(int)
    regime_efficiency = high_vol_efficiency - low_vol_efficiency
    
    # Volume-Scaled Extreme Reversal
    # Abnormal move identification
    recent_high = df['high'].rolling(5).max()
    recent_low = df['low'].rolling(5).min()
    lookback_high = df['high'].rolling(20).max()
    lookback_low = df['low'].rolling(20).min()
    
    near_high = (recent_high - lookback_high) / lookback_high
    near_low = (recent_low - lookback_low) / lookback_low
    
    return_deviation = df['close'].pct_change() - df['close'].pct_change().rolling(10).mean()
    
    # Volume context
    volume_ratio = df['volume'] / df['volume'].rolling(10).mean()
    volume_clustering = (df['volume'] > df['volume'].rolling(5).mean()).rolling(3).sum()
    
    # Reversal signal
    high_reversal = -near_high * volume_ratio * (return_deviation < 0).astype(int)
    low_reversal = near_low * volume_ratio * (return_deviation > 0).astype(int)
    reversal_signal = high_reversal + low_reversal
    
    # Amount-Based Flow Persistence
    # Directional order flow
    up_day = df['close'] > df['close'].shift(1)
    down_day = df['close'] < df['close'].shift(1)
    
    up_amount = df['amount'] * up_day.astype(int)
    down_amount = df['amount'] * down_day.astype(int)
    net_flow = (up_amount - down_amount) / df['amount'].rolling(20).mean()
    
    # Flow persistence
    flow_direction = (net_flow > 0).astype(int) - (net_flow < 0).astype(int)
    consecutive_flow = flow_direction.groupby(
        (flow_direction != flow_direction.shift()).cumsum()
    ).cumcount() + 1
    flow_persistence = consecutive_flow * np.sign(net_flow)
    
    # Flow momentum
    flow_momentum = net_flow.diff(3)
    
    # Combined flow signal
    flow_signal = flow_persistence * flow_momentum
    
    # Final alpha factor combination
    alpha_factor = (
        0.3 * pv_divergence.rank(pct=True) +
        0.25 * regime_efficiency.rank(pct=True) +
        0.25 * reversal_signal.rank(pct=True) +
        0.2 * flow_signal.rank(pct=True)
    )
    
    return alpha_factor
