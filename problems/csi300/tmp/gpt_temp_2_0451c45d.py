import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate novel alpha factor combining intraday regime transitions, asymmetric range dynamics,
    volume-volatility regime switching, gap momentum persistence, transaction value efficiency,
    and price-volume-timing harmony.
    """
    # Make copy to avoid modifying original data
    data = df.copy()
    
    # 1. Intraday Regime Transition
    # Morning volatility clustering (High-Open vs Open-Low)
    morning_vol_cluster = (data['high'] - data['open']) / (data['open'] - data['low'] + 1e-8)
    morning_vol_cluster = morning_vol_cluster.replace([np.inf, -np.inf], 0)
    
    # Afternoon mean reversion intensity (Close-Low vs High-Close)
    afternoon_reversion = (data['close'] - data['low']) / (data['high'] - data['close'] + 1e-8)
    afternoon_reversion = afternoon_reversion.replace([np.inf, -np.inf], 0)
    
    # 2. Asymmetric Range Dynamics
    # Upper range efficiency (High-Open vs High-Low)
    upper_range_eff = (data['high'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # Lower range defense (Open-Low vs High-Low)
    lower_range_def = (data['open'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    
    # 3. Volume-Volatility Regime Switching
    # Calculate daily volatility (range-based)
    daily_volatility = (data['high'] - data['low']) / data['open']
    
    # Volume concentration in high-volatility periods
    vol_vol_regime = data['volume'] * daily_volatility
    
    # Low-volatility volume accumulation divergence
    low_vol_volume = data['volume'] / (daily_volatility + 1e-8)
    
    # 4. Gap Momentum Persistence
    # Gap direction consistency (today's open vs yesterday's close)
    gap_direction = data['open'] / data['close'].shift(1) - 1
    
    # Gap size vs subsequent intraday range
    gap_vs_range = (data['open'] - data['close'].shift(1)).abs() / (data['high'] - data['low'] + 1e-8)
    
    # 5. Transaction Value Efficiency
    # Amount per volume unit vs price level
    transaction_efficiency = data['amount'] / (data['volume'] * data['close'] + 1e-8)
    
    # Large transaction concentration patterns (using rolling quantile)
    large_trans_ratio = data['amount'].rolling(window=5).apply(
        lambda x: (x > x.quantile(0.8)).sum() / len(x) if len(x) == 5 else np.nan
    )
    
    # 6. Price-Volume-Timing Harmony
    # Volume timing relative to price extremes
    # Higher volume when price is near extremes is more significant
    price_position = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    volume_timing = data['volume'] * (1 - 2 * abs(price_position - 0.5))
    
    # Intraday volume distribution efficiency
    # Normalize volume by range to get volume efficiency
    volume_efficiency = data['volume'] / (data['high'] - data['low'] + 1e-8)
    
    # Combine all components with appropriate weights and normalization
    factors = pd.DataFrame({
        'morning_vol': morning_vol_cluster,
        'afternoon_rev': afternoon_reversion,
        'upper_eff': upper_range_eff,
        'lower_def': lower_range_def,
        'vol_vol': vol_vol_regime,
        'low_vol_vol': low_vol_volume,
        'gap_dir': gap_direction,
        'gap_range': gap_vs_range,
        'trans_eff': transaction_efficiency,
        'large_trans': large_trans_ratio,
        'vol_timing': volume_timing,
        'vol_eff': volume_efficiency
    })
    
    # Z-score normalization for each component
    factors_normalized = factors.apply(lambda x: (x - x.rolling(window=20).mean()) / (x.rolling(window=20).std() + 1e-8))
    
    # Final alpha factor - weighted combination
    weights = {
        'morning_vol': 0.08,
        'afternoon_rev': 0.08,
        'upper_eff': 0.07,
        'lower_def': 0.07,
        'vol_vol': 0.10,
        'low_vol_vol': 0.10,
        'gap_dir': 0.08,
        'gap_range': 0.08,
        'trans_eff': 0.12,
        'large_trans': 0.10,
        'vol_timing': 0.06,
        'vol_eff': 0.06
    }
    
    alpha_factor = pd.Series(0, index=data.index)
    for col, weight in weights.items():
        alpha_factor += factors_normalized[col] * weight
    
    # Final smoothing and outlier handling
    alpha_factor = alpha_factor.rolling(window=3).mean()
    alpha_factor = alpha_factor.clip(lower=-3, upper=3)
    
    return alpha_factor
