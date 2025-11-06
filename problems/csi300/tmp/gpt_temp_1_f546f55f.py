import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor using multiple heuristic components including:
    - Price-Volume Divergence Momentum
    - Volatility-Scaled Range Efficiency  
    - Volume-Confirmed Extreme Reversal
    - Amount Flow Regime Detection
    - Volatility-Volume Regime Composite
    """
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Extract price and volume data
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    amount = df['amount']
    
    # Price-Volume Divergence Momentum Components
    mom_5d = close / close.shift(5) - 1
    mom_10d = close / close.shift(10) - 1
    mom_ratio = (close / close.shift(5)) / (close.shift(5) / close.shift(10))
    
    vol_trend_5d = volume / volume.shift(5)
    vol_accel = (volume / volume.shift(5)) / (volume.shift(5) / volume.shift(10))
    
    # Volume persistence: count of days where volume > previous day's volume
    vol_persistence = pd.Series(index=df.index, dtype=float)
    for i in range(5, len(df)):
        vol_persistence.iloc[i] = (volume.iloc[i-4:i+1] > volume.shift(1).iloc[i-4:i+1]).sum() / 5
    
    # Divergence factors
    pos_divergence = ((mom_5d > 0) & (vol_trend_5d < 0.8)).astype(float)
    neg_divergence = ((mom_5d < 0) & (vol_trend_5d > 1.2)).astype(float)
    mom_vol_composite = mom_5d * vol_trend_5d
    
    # Volatility-Scaled Range Efficiency Components
    # 5-day volatility: (max high - min low) / close_5d_ago
    vol_5d = pd.Series(index=df.index, dtype=float)
    for i in range(5, len(df)):
        vol_5d.iloc[i] = (high.iloc[i-4:i+1].max() - low.iloc[i-4:i+1].min()) / close.iloc[i-5]
    
    # Daily efficiency
    daily_eff = abs(close - close.shift(1)) / (high - low)
    daily_eff = daily_eff.replace([np.inf, -np.inf], np.nan)
    
    # Gap efficiency
    gap_eff = abs(close - close.shift(1)) / (np.maximum(high, close.shift(1)) - np.minimum(low, close.shift(1)))
    gap_eff = gap_eff.replace([np.inf, -np.inf], np.nan)
    
    # Multi-period efficiency
    mom_3d = abs(close / close.shift(3) - 1)
    
    range_eff_3d = pd.Series(index=df.index, dtype=float)
    for i in range(3, len(df)):
        price_change = abs(close.iloc[i] / close.iloc[i-3] - 1)
        price_range = (high.iloc[i-2:i+1].max() - low.iloc[i-2:i+1].min()) / close.iloc[i-3]
        range_eff_3d.iloc[i] = price_change / price_range if price_range > 0 else 0
    
    # Efficiency persistence
    eff_persistence = pd.Series(index=df.index, dtype=float)
    for i in range(5, len(df)):
        eff_persistence.iloc[i] = (daily_eff.iloc[i-4:i+1] > 0.7).sum() / 5
    
    # Volatility-scaled factors
    mom_eff = mom_5d / vol_5d
    mom_eff = mom_eff.replace([np.inf, -np.inf], np.nan)
    
    range_breakout = (high - low) / vol_5d
    range_breakout = range_breakout.replace([np.inf, -np.inf], np.nan)
    
    eff_mom = range_eff_3d - range_eff_3d.shift(5)
    
    # Volume-Confirmed Extreme Reversal Components
    price_dev_3d = pd.Series(index=df.index, dtype=float)
    for i in range(3, len(df)):
        price_dev_3d.iloc[i] = (close.iloc[i] - close.iloc[i-3]) / (high.iloc[i-2:i+1].max() - low.iloc[i-2:i+1].min())
    
    vol_spike = volume / ((volume.shift(1) + volume.shift(2) + volume.shift(3)) / 3)
    
    abnormal_range = pd.Series(index=df.index, dtype=float)
    for i in range(3, len(df)):
        abnormal_range.iloc[i] = (high.iloc[i] - low.iloc[i]) / ((high.iloc[i-1] - low.iloc[i-1] + 
                                                                high.iloc[i-2] - low.iloc[i-2] + 
                                                                high.iloc[i-3] - low.iloc[i-3]) / 3)
    
    # Multi-timeframe reversal signals
    short_term_rev = ((price_dev_3d < -0.3) & (vol_spike > 1.5)).astype(float)
    medium_term_rev = ((mom_5d < -0.05) & (vol_trend_5d > 1.2)).astype(float)
    vol_weighted_rev = (close / close.shift(1) - 1) * (volume / volume.shift(1))
    
    # Amount Flow Regime Detection
    # Directional flow analysis
    up_day_amount = amount.where(close > close.shift(1), 0)
    down_day_amount = amount.where(close < close.shift(1), 0)
    net_flow = np.sign(close - close.shift(1)) * amount
    
    # Flow momentum patterns
    net_flow_3d = pd.Series(index=df.index, dtype=float)
    for i in range(3, len(df)):
        net_flow_3d.iloc[i] = net_flow.iloc[i-2:i+1].sum()
    
    # Flow consistency
    flow_consistency = pd.Series(index=df.index, dtype=float)
    for i in range(5, len(df)):
        flow_directions = np.sign(close.iloc[i-4:i+1] - close.shift(1).iloc[i-4:i+1])
        flow_consistency.iloc[i] = (flow_directions == flow_directions.iloc[0]).sum() / 5
    
    flow_accel = (amount / amount.shift(3)) / (amount.shift(3) / amount.shift(6))
    flow_accel = flow_accel.replace([np.inf, -np.inf], np.nan)
    
    # Regime-based signals
    persistent_buying = ((flow_consistency > 0.7) & (net_flow_3d > 0)).astype(float)
    
    # Selling exhaustion: down_day_amount decreasing over 3 days
    selling_exhaustion = pd.Series(index=df.index, dtype=float)
    for i in range(3, len(df)):
        down_amounts = down_day_amount.iloc[i-2:i+1]
        selling_exhaustion.iloc[i] = (down_amounts.iloc[0] > down_amounts.iloc[1] > down_amounts.iloc[2]).astype(float)
    
    flow_mom_divergence = ((net_flow_3d > 0) & (mom_5d < 0)).astype(float)
    
    # Volatility-Volume Regime Composite
    # Regime classification
    high_vol_regime = (vol_5d > 0.03).astype(float)
    low_vol_regime = (vol_5d < 0.015).astype(float)
    transition_regime = ((vol_5d >= 0.015) & (vol_5d <= 0.03)).astype(float)
    
    # Volume pattern by regime
    vol_persistence_high_vol = pd.Series(index=df.index, dtype=float)
    vol_median_10d = volume.rolling(window=10, min_periods=1).median()
    
    for i in range(5, len(df)):
        if high_vol_regime.iloc[i]:
            vol_persistence_high_vol.iloc[i] = (volume.iloc[i-4:i+1] > volume.shift(1).iloc[i-4:i+1]).sum()
    
    low_vol_spike = volume / vol_median_10d
    
    # Adaptive alpha factors
    high_vol_continuation = (close / close.shift(1) - 1) * vol_persistence_high_vol * high_vol_regime
    low_vol_reversal = -1 * (close / close.shift(1) - 1) * low_vol_spike * low_vol_regime
    
    vol_change = vol_5d / vol_5d.shift(1) - 1
    vol_change = vol_change.replace([np.inf, -np.inf], np.nan)
    
    vol_vol_change = volume / volume.shift(1) - 1
    regime_transition = vol_change * vol_vol_change * transition_regime
    
    # Composite factor construction
    # Weight different components based on their predictive power
    composite_factor = (
        # Price-Volume Divergence (30%)
        0.3 * (pos_divergence - neg_divergence + mom_vol_composite) +
        # Volatility Efficiency (25%)
        0.25 * (mom_eff + range_breakout + eff_mom) +
        # Extreme Reversal (20%)
        0.2 * (short_term_rev + medium_term_rev + vol_weighted_rev) +
        # Flow Regime (15%)
        0.15 * (persistent_buying + selling_exhaustion - flow_mom_divergence) +
        # Vol-Volume Regime (10%)
        0.1 * (high_vol_continuation + low_vol_reversal + regime_transition)
    )
    
    # Normalize the composite factor
    result = (composite_factor - composite_factor.rolling(window=20, min_periods=1).mean()) / composite_factor.rolling(window=20, min_periods=1).std()
    
    return result
