import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Aware Price Discovery Efficiency with Volume Flow Analysis
    """
    data = df.copy()
    
    # Market Regime Classification
    # Opening efficiency
    data['open_eff'] = np.abs(data['open'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1))
    data['open_eff'] = data['open_eff'].replace([np.inf, -np.inf], np.nan)
    
    # Closing efficiency
    data['close_eff'] = np.abs(data['close'] - (data['high'] + data['low']) / 2) / (data['high'] - data['low'])
    data['close_eff'] = data['close_eff'].replace([np.inf, -np.inf], np.nan)
    
    # Regime classification based on efficiency patterns
    data['eff_ratio'] = data['open_eff'] / data['close_eff']
    data['eff_ratio'] = data['eff_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # High efficiency regime: both open and close efficiency are low
    high_eff_threshold = data['open_eff'].rolling(window=20, min_periods=10).quantile(0.3)
    data['high_eff_regime'] = (data['open_eff'] < high_eff_threshold) & (data['close_eff'] < data['close_eff'].rolling(window=20, min_periods=10).quantile(0.3))
    
    # Low efficiency regime: high opening inefficiency
    low_eff_threshold = data['open_eff'].rolling(window=20, min_periods=10).quantile(0.7)
    data['low_eff_regime'] = data['open_eff'] > low_eff_threshold
    
    # Normal regime: everything else
    data['normal_regime'] = ~data['high_eff_regime'] & ~data['low_eff_regime']
    
    # Volume Flow Direction Analysis
    # Volume-weighted directional flow
    price_change = data['close'] - data['open']
    data['vol_directional_flow'] = (price_change / (data['high'] - data['low']).replace(0, np.nan)) * data['volume']
    data['vol_directional_flow'] = data['vol_directional_flow'].replace([np.inf, -np.inf], np.nan)
    
    # Volume flow persistence (3-day)
    data['vol_flow_persistence'] = data['vol_directional_flow'].rolling(window=3, min_periods=2).apply(
        lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 and not np.isnan(x).any() else 0
    )
    
    # Price-volume divergence
    price_trend = data['close'].rolling(window=5, min_periods=3).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0
    )
    volume_trend = data['volume'].rolling(window=5, min_periods=3).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0
    )
    data['price_volume_divergence'] = np.sign(price_trend) != np.sign(volume_trend)
    
    # Multi-timeframe Order Imbalance
    # Opening gap and volume intensity
    data['opening_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_volume_intensity'] = data['opening_gap'] * (data['volume'] / data['volume'].rolling(window=20, min_periods=10).mean())
    
    # Intraday volume distribution
    high_low_range = data['high'] - data['low']
    close_to_open_range = np.abs(data['close'] - data['open'])
    data['intraday_vol_concentration'] = close_to_open_range / high_low_range.replace(0, np.nan)
    data['intraday_vol_concentration'] = data['intraday_vol_concentration'].replace([np.inf, -np.inf], np.nan)
    
    # Closing auction pressure
    data['closing_pressure'] = (data['close'] - (data['high'] + data['low']) / 2) / high_low_range.replace(0, np.nan)
    data['closing_pressure'] = data['closing_pressure'].replace([np.inf, -np.inf], np.nan)
    
    # Order imbalance convergence score
    data['order_imbalance_score'] = (
        data['gap_volume_intensity'].rolling(window=5, min_periods=3).mean() +
        data['intraday_vol_concentration'].rolling(window=5, min_periods=3).mean() +
        data['closing_pressure'].rolling(window=5, min_periods=3).mean()
    ) / 3
    
    # Price Discovery Latency Factor
    # Price adjustment delay metrics
    data['intraday_return'] = (data['close'] - data['open']) / data['open']
    data['overnight_return'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Discovery latency score (how quickly price adjusts to overnight information)
    data['discovery_latency'] = np.abs(data['intraday_return']) / (np.abs(data['overnight_return']) + 1e-8)
    data['discovery_latency'] = data['discovery_latency'].replace([np.inf, -np.inf], np.nan)
    
    # Latency-return relationship
    data['latency_return_correlation'] = data['discovery_latency'].rolling(window=10, min_periods=5).corr(data['intraday_return'])
    
    # Adaptive Composite Discovery Factor
    # Regime-classified signal weighting
    high_eff_signal = (
        -data['discovery_latency'].rolling(window=5, min_periods=3).mean() +  # Lower latency preferred in efficient markets
        data['vol_flow_persistence'] +  # Consistent volume flow
        -data['price_volume_divergence'].astype(float)  # Avoid divergence
    )
    
    low_eff_signal = (
        data['order_imbalance_score'] +  # Order imbalance matters more
        data['vol_directional_flow'].rolling(window=3, min_periods=2).mean() +  # Recent volume flow
        -data['discovery_latency']  # But still prefer lower latency
    )
    
    normal_signal = (
        data['vol_flow_persistence'] +
        data['order_imbalance_score'] +
        -data['discovery_latency'] +
        -data['price_volume_divergence'].astype(float)
    ) / 4
    
    # Apply regime-aware weighting
    regime_weighted_signal = (
        data['high_eff_regime'].astype(float) * high_eff_signal +
        data['low_eff_regime'].astype(float) * low_eff_signal +
        data['normal_regime'].astype(float) * normal_signal
    )
    
    # Volume flow confirmation
    volume_confirmation = data['vol_directional_flow'].rolling(window=3, min_periods=2).std()
    volume_confirmation = 1 / (1 + volume_confirmation)  # Prefer stable volume flow
    
    # Multi-timeframe alignment
    short_term_signal = regime_weighted_signal.rolling(window=3, min_periods=2).mean()
    medium_term_signal = regime_weighted_signal.rolling(window=5, min_periods=3).mean()
    
    # Final regime-adaptive discovery efficiency factor
    discovery_factor = (
        short_term_signal * 0.4 +
        medium_term_signal * 0.4 +
        volume_confirmation * 0.2
    )
    
    # Normalize and clean
    discovery_factor = (discovery_factor - discovery_factor.rolling(window=20, min_periods=10).mean()) / discovery_factor.rolling(window=20, min_periods=10).std()
    discovery_factor = discovery_factor.replace([np.inf, -np.inf], np.nan)
    
    return discovery_factor
