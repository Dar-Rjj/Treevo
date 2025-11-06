import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Regime Momentum factor combining multiple timeframes, volume confirmation,
    range efficiency, extreme move detection, directional flow, and volatility adaptation.
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Multi-Timeframe Momentum
    data['mom_5d'] = data['close'] / data['close'].shift(5) - 1
    data['mom_10d'] = data['close'] / data['close'].shift(10) - 1
    data['mom_accel'] = (data['close'] / data['close'].shift(5)) / (data['close'].shift(5) / data['close'].shift(10))
    
    # Volume Confirmation
    data['vol_trend'] = data['volume'] / data['volume'].shift(5)
    data['vol_mom'] = (data['volume'] / data['volume'].shift(5)) / (data['volume'].shift(5) / data['volume'].shift(10))
    data['vol_price_corr'] = np.sign(data['close'] - data['close'].shift(1)) * (data['volume'] / data['volume'].shift(1))
    
    # Range Efficiency Components
    data['daily_range'] = data['high'] - data['low']
    data['gap_adj_range'] = np.maximum(data['high'], data['close'].shift(1)) - np.minimum(data['low'], data['close'].shift(1))
    data['norm_range'] = (data['high'] - data['low']) / data['close'].shift(1)
    
    # Efficiency Metrics
    data['single_eff'] = np.abs(data['close'] - data['close'].shift(1)) / (data['high'] - data['low']).replace(0, np.nan)
    data['cum_range_3d'] = data['daily_range'].rolling(window=3).sum()
    data['eff_3d'] = np.abs(data['close'] - data['close'].shift(3)) / data['cum_range_3d'].replace(0, np.nan)
    data['eff_persistence'] = (data['single_eff'] > 0.7).rolling(window=5).sum()
    
    # Extreme Move Detection
    data['price_zscore'] = (data['close'] - data['close'].rolling(window=3).mean()) / data['close'].rolling(window=3).std()
    data['range_expansion'] = data['daily_range'] / data['daily_range'].rolling(window=5).mean()
    mom_ratio = data['close'] / data['close'].shift(1)
    data['mom_extreme'] = mom_ratio / mom_ratio.rolling(window=5).std()
    
    # Volume Confirmation for Extremes
    data['vol_zscore'] = data['volume'] / data['volume'].rolling(window=5).mean()
    data['vol_spike_persistence'] = (data['volume'] > 2 * data['volume'].rolling(window=5).mean()).rolling(window=3).sum()
    data['vol_price_divergence'] = (data['volume'] / data['volume'].shift(1)) / (data['close'] / data['close'].shift(1))
    
    # Directional Amount Flow
    up_mask = data['close'] > data['close'].shift(1)
    down_mask = data['close'] < data['close'].shift(1)
    
    data['up_flow'] = np.where(up_mask, data['amount'], 0)
    data['down_flow'] = np.where(down_mask, data['amount'], 0)
    data['net_flow_3d'] = (data['up_flow'] - data['down_flow']).rolling(window=3).sum()
    
    # Flow momentum and consistency
    data['flow_dir_consistency'] = np.sign(data['close'] - data['close'].shift(1)).rolling(window=5).apply(lambda x: len(set(x.dropna())) if len(x.dropna()) > 0 else np.nan)
    prev_net_flow = (data['up_flow'].shift(3) - data['down_flow'].shift(3)).rolling(window=3).sum()
    data['flow_accel'] = (data['net_flow_3d'] / prev_net_flow.replace(0, np.nan)) - 1
    
    # Volatility Regime
    data['range_vol'] = (data['high'].rolling(window=10).max() - data['low'].rolling(window=10).min()) / data['close'].shift(10)
    data['vol_ratio'] = data['close'].rolling(window=5).std() / data['close'].shift(5).rolling(window=5).std()
    
    # Volume patterns by regime
    high_vol_threshold = data['range_vol'].quantile(0.7)
    low_vol_threshold = data['range_vol'].quantile(0.3)
    
    data['vol_regime_adj'] = np.where(
        data['range_vol'] > high_vol_threshold,
        data['volume'] / data['volume'].rolling(window=10).median(),
        data['volume'] / data['volume'].rolling(window=10).mean()
    )
    
    # Composite factor calculation
    # 1. Price-Volume Regime Momentum
    regime_momentum = (
        data['mom_5d'] * 0.4 + 
        data['mom_10d'] * 0.3 + 
        data['mom_accel'] * 0.3
    ) * np.where(
        (data['mom_5d'] > 0.05) & (data['vol_trend'] < 0.9), -1,  # High momentum + declining volume = reversal
        np.where(
            (data['mom_5d'] < -0.02) & (data['vol_trend'] > 1.5), 1,  # Low momentum + volume surge = breakout
            data['vol_price_corr']  # Otherwise use volume-price correlation
        )
    )
    
    # 2. Range Efficiency with Volatility Scaling
    efficiency_signal = (
        data['single_eff'] * 0.5 + 
        data['eff_3d'] * 0.3 + 
        data['eff_persistence'] * 0.2
    ) * np.where(
        (data['single_eff'] > 0.8) & (data['range_vol'] < data['range_vol'].quantile(0.3)), 1,  # High efficiency + low vol = trend quality
        np.where(
            (data['single_eff'] < 0.3) & (data['range_vol'] > data['range_vol'].quantile(0.7)), -1,  # Low efficiency + high vol = noise
            np.sign(data['eff_3d'] - data['eff_3d'].shift(1))  # Efficiency improvement direction
        )
    )
    
    # 3. Extreme Move Reversal
    extreme_signal = np.where(
        (np.abs(data['price_zscore']) > 2) & (np.abs(data['vol_zscore']) < 1.5), 
        -np.sign(data['price_zscore']),  # Extreme price + normal volume = mean reversion
        np.where(
            (np.abs(data['price_zscore']) > 2) & (np.abs(data['vol_zscore']) > 2),
            np.sign(data['price_zscore']),  # Extreme price + extreme volume = momentum
            np.sign(data['vol_price_divergence'])  # Volume precedes price
        )
    )
    
    # 4. Directional Flow Persistence
    flow_signal = (
        data['net_flow_3d'] * 0.6 + 
        data['flow_accel'].fillna(0) * 0.4
    ) * np.where(
        data['flow_dir_consistency'] == 1, 1.2,  # Strong directional persistence
        np.where(
            (np.abs(data['net_flow_3d']) > data['net_flow_3d'].rolling(window=20).quantile(0.8)) & 
            (np.abs(data['mom_5d']) < 0.02), 1.5,  # High flow + small price moves
            1.0  # Normal confirmation
        )
    )
    
    # 5. Volatility-Regime Adaptive Weighting
    vol_regime_weight = np.where(
        (data['range_vol'] > high_vol_threshold) & (data['vol_regime_adj'] > 1.2), 0.4,  # High vol + high volume = momentum focus
        np.where(
            (data['range_vol'] < low_vol_threshold) & (data['vol_regime_adj'] > 1.5), 0.6,  # Low vol + volume spike = breakout focus
            np.where(
                (data['vol_ratio'] > 1.5) & (data['vol_trend'] < 0.8), 0.2,  # Vol expansion + volume decline = caution
                0.3  # Normal regime
            )
        )
    )
    
    # Final composite factor
    composite_factor = (
        regime_momentum * 0.25 +
        efficiency_signal * 0.20 +
        extreme_signal * 0.25 +
        flow_signal * 0.30
    ) * vol_regime_weight
    
    # Normalize and return
    factor = (composite_factor - composite_factor.rolling(window=20).mean()) / composite_factor.rolling(window=20).std()
    
    return factor
