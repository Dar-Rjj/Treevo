import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Generate alpha factor combining volatility-normalized price-volume divergence,
    regime-aware range efficiency, volume-scaled mean reversion, and amount-based flow persistence.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    alpha = pd.Series(index=data.index, dtype=float)
    
    # 1. Volatility-Normalized Price-Volume Divergence
    # Multi-Timeframe Price Momentum
    short_momentum = data['close'].pct_change(3)
    medium_momentum = data['close'].pct_change(10)
    momentum_divergence = short_momentum - medium_momentum
    
    # Volume Persistence Analysis
    volume_trend = data['volume'].rolling(window=5).apply(
        lambda x: stats.linregress(range(len(x)), x)[0], raw=False
    )
    volume_acceleration = volume_trend.diff()
    
    # Volume regime detection
    volume_ma = data['volume'].rolling(window=10).mean()
    high_volume_regime = (data['volume'] > volume_ma * 1.2).astype(int)
    
    # Divergence Signal Construction
    price_volatility = data['close'].pct_change().rolling(window=20).std()
    normalized_momentum = momentum_divergence / (price_volatility + 1e-8)
    
    # Compare price momentum direction with volume trend direction
    price_direction = np.sign(short_momentum)
    volume_direction = np.sign(volume_trend)
    alignment_score = (price_direction == volume_direction).astype(int) - 0.5
    
    divergence_factor = normalized_momentum * alignment_score * (1 + high_volume_regime)
    
    # 2. Regime-Aware Range Efficiency
    # True Range Calculation
    prev_close = data['close'].shift(1)
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - prev_close)
    tr3 = abs(data['low'] - prev_close)
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = true_range.rolling(window=5).mean()
    
    # Price Movement Efficiency
    abs_return = abs(data['close'] - prev_close)
    abs_efficiency = abs_return / (true_range + 1e-8)
    directional_efficiency = (data['close'] - prev_close) / (true_range + 1e-8)
    
    # Regime Classification
    tr_median = true_range.rolling(window=20).median()
    high_vol_regime = (true_range > tr_median).astype(int)
    low_vol_regime = (true_range <= tr_median).astype(int)
    
    # Regime-specific efficiency
    high_vol_efficiency = abs_efficiency * high_vol_regime
    low_vol_efficiency = directional_efficiency * low_vol_regime
    range_efficiency_factor = high_vol_efficiency + low_vol_efficiency
    
    # 3. Volume-Scaled Mean Reversion
    # Extreme Move Identification
    price_range_20d = data['high'].rolling(window=20).max() - data['low'].rolling(window=20).min()
    recent_high_rel = (data['high'].rolling(window=5).max() - data['low'].rolling(window=20).min()) / price_range_20d
    recent_low_rel = (data['high'].rolling(window=20).max() - data['low'].rolling(window=5).min()) / price_range_20d
    
    return_zscore = data['close'].pct_change().rolling(window=20).apply(
        lambda x: (x[-1] - x.mean()) / (x.std() + 1e-8), raw=True
    )
    
    # Oversold/overbought conditions
    overbought = (recent_high_rel > 0.8) & (return_zscore > 1.5)
    oversold = (recent_low_rel < 0.2) & (return_zscore < -1.5)
    
    # Volume Confirmation
    volume_ratio = data['volume'] / data['volume'].rolling(window=5).mean()
    volume_trend_confirmation = (np.sign(data['close'].diff()) == np.sign(volume_trend)).astype(int)
    
    # Reversal Signal Generation
    reversal_strength = np.where(overbought, -1, np.where(oversold, 1, 0))
    volume_scaled_reversal = reversal_strength * volume_ratio * volume_trend_confirmation
    
    # 4. Amount-Based Flow Persistence
    # Directional Order Flow
    up_day_mask = data['close'] > prev_close
    down_day_mask = data['close'] < prev_close
    
    up_day_amount = data['amount'].where(up_day_mask, 0)
    down_day_amount = data['amount'].where(down_day_mask, 0)
    
    net_flow = up_day_amount - down_day_amount
    net_flow_ma = net_flow.rolling(window=5).mean()
    
    # Flow Momentum
    flow_direction = np.sign(net_flow)
    flow_direction_shift = flow_direction.shift(1)
    consecutive_flow_days = (flow_direction == flow_direction_shift).astype(int)
    consecutive_flow_days = consecutive_flow_days.groupby(consecutive_flow_days.ne(consecutive_flow_days.shift()).cumsum()).cumcount() + 1
    
    flow_acceleration = net_flow.diff()
    
    # Regime-Adaptive Flow Signals
    flow_to_volume = net_flow / (data['volume'] + 1e-8)
    high_flow_regime = (abs(net_flow) > abs(net_flow_ma) * 1.5).astype(int)
    
    flow_persistence = consecutive_flow_days * np.sign(net_flow) * (1 + high_flow_regime)
    flow_factor = flow_persistence * flow_to_volume
    
    # Combine all factors with equal weights
    combined_alpha = (
        0.25 * divergence_factor.fillna(0) +
        0.25 * range_efficiency_factor.fillna(0) +
        0.25 * volume_scaled_reversal.fillna(0) +
        0.25 * flow_factor.fillna(0)
    )
    
    # Normalize the final alpha
    alpha = (combined_alpha - combined_alpha.rolling(window=20).mean()) / (combined_alpha.rolling(window=20).std() + 1e-8)
    
    return alpha
