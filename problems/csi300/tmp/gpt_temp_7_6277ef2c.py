import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic price returns for various periods
    close = df['close']
    high = df['high']
    low = df['low']
    open_price = df['open']
    volume = df['volume']
    amount = df['amount']
    
    # 1. Regime-Adaptive Momentum Acceleration
    # Multi-timeframe acceleration alignment
    ret_3d = close / close.shift(3) - 1
    ret_6d = close / close.shift(6) - 1
    ret_5d = close / close.shift(5) - 1
    ret_10d = close / close.shift(10) - 1
    
    short_term_accel = ret_3d - ret_6d
    medium_term_accel = ret_5d / (ret_10d + 1e-8)
    
    # VWAP calculation
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).rolling(20).sum() / volume.rolling(20).sum()
    vwap_accel = (close - vwap) / vwap
    
    # Volatility regime classification
    high_low_range = (high - low) / close
    vol_regime = high_low_range.rolling(20).apply(lambda x: (x > x.quantile(0.7)).sum() / len(x), raw=False)
    
    # Volume trend participation
    vol_5d = volume.rolling(5).sum()
    vol_20d = volume.rolling(20).sum()
    vol_trend = vol_5d / vol_20d
    
    regime_adaptive_momentum = (short_term_accel + medium_term_accel + vwap_accel) * (1 + vol_regime) * vol_trend
    
    # 2. Price-Level Volume Elasticity with Liquidity
    # Relative price position analysis
    high_20d = high.rolling(20).max()
    low_20d = low.rolling(20).min()
    price_position = (close - low_20d) / (high_20d - low_20d + 1e-8)
    
    # Volume shock intensity
    vol_20d_avg = volume.rolling(20).mean()
    vol_shock = volume / vol_20d_avg
    vol_change_5d = volume / volume.shift(5)
    
    # Dollar volume pressure analysis
    avg_amount_20d = amount.rolling(20).mean()
    liquidity_ratio = amount / avg_amount_20d
    
    price_volume_elasticity = price_position * vol_shock * vol_change_5d * liquidity_ratio
    
    # 3. Intraday Efficiency Momentum Enhancement
    # Intraday strength efficiency
    intraday_efficiency = (close - open_price) / (high - low + 1e-8)
    
    # Overnight gap reversion analysis
    overnight_gap = (open_price - close.shift(1)) / close.shift(1)
    gap_vs_range = abs(overnight_gap) / (high - low + 1e-8)
    reversion_prob = 1 - gap_vs_range
    
    # Liquidity-filtered efficiency
    liquidity_filter = (liquidity_ratio > 1.2).astype(float)
    intraday_momentum = intraday_efficiency * reversion_prob * liquidity_filter
    
    # 4. Liquidity-Clustered Momentum Decay Patterns
    # Multi-period momentum decay analysis
    mom_3d = close / close.shift(3) - 1
    mom_6d = close / close.shift(6) - 1
    mom_5d = close / close.shift(5) - 1
    mom_10d = close / close.shift(10) - 1
    
    mom_decay_3_6 = mom_3d - mom_6d
    mom_decay_5_10 = mom_5d / (mom_10d + 1e-8)
    
    # Volume trend during momentum phases
    vol_during_mom = volume.rolling(5).mean() / volume.rolling(20).mean()
    
    # Liquidity-enhanced decay signals
    liquidity_decay = (mom_decay_3_6 + mom_decay_5_10) * vol_during_mom * liquidity_ratio
    
    # 5. Volume-Price Divergence Synthesis with Acceleration
    # Momentum-volume divergence analysis
    accel_vol_divergence = short_term_accel - vol_trend
    
    # Price range efficiency divergence
    true_range = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1).max(axis=1)
    
    price_efficiency = (close - close.shift(1)) / (true_range + 1e-8)
    efficiency_divergence = price_efficiency - vol_shock
    
    # Combined divergence signals
    combined_divergence = (accel_vol_divergence + efficiency_divergence) * liquidity_ratio
    
    # Final factor synthesis
    factor = (
        0.25 * regime_adaptive_momentum +
        0.20 * price_volume_elasticity +
        0.20 * intraday_momentum +
        0.15 * liquidity_decay +
        0.20 * combined_divergence
    )
    
    # Ensure no future data is used
    result = factor
    
    return result
