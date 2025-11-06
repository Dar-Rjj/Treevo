import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility Regime-Adaptive Price-Volume Divergence with Liquidity Stress Testing
    """
    data = df.copy()
    
    # Volatility Regime Classification
    # Multi-Scale Volatility Assessment
    data['intraday_vol'] = (data['high'] - data['low']) / data['close']
    
    # 3-day Average True Range
    data['tr'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['atr_3d'] = data['tr'].rolling(window=3, min_periods=2).mean()
    data['vol_ratio'] = data['intraday_vol'] / (data['atr_3d'] / data['close'].shift(1))
    
    # Medium-Term Volatility Structure
    data['vol_5d'] = data['intraday_vol'].rolling(window=5, min_periods=3).mean()
    data['vol_cluster'] = (data['intraday_vol'] > data['intraday_vol'].rolling(window=5, min_periods=3).mean()).rolling(window=3).sum()
    
    # Volatility regime boundaries
    vol_20d_median = data['intraday_vol'].rolling(window=20, min_periods=10).median()
    high_vol_threshold = vol_20d_median * 1.5
    low_vol_threshold = vol_20d_median * 0.7
    
    data['vol_regime'] = 1  # Normal
    data.loc[data['intraday_vol'] > high_vol_threshold, 'vol_regime'] = 2  # High
    data.loc[data['intraday_vol'] < low_vol_threshold, 'vol_regime'] = 0  # Low
    
    # Price-Volume Divergence Core
    # Directional Volume Analysis
    data['price_change'] = data['close'].pct_change()
    data['up_day'] = (data['price_change'] > 0).astype(int)
    data['down_day'] = (data['price_change'] < 0).astype(int)
    
    # Up-day vs Down-day volume patterns
    up_day_volume = data['volume'].where(data['up_day'] == 1)
    down_day_volume = data['volume'].where(data['down_day'] == 1)
    
    data['avg_up_volume'] = up_day_volume.rolling(window=10, min_periods=5).mean()
    data['avg_down_volume'] = down_day_volume.rolling(window=10, min_periods=5).mean()
    
    # Volume-Price Momentum Divergence
    data['volume_change'] = data['volume'].pct_change()
    data['price_momentum'] = data['price_change'].rolling(window=3, min_periods=2).mean()
    data['volume_momentum'] = data['volume_change'].rolling(window=3, min_periods=2).mean()
    
    data['price_vol_divergence'] = np.where(
        (data['price_momentum'] > 0) & (data['volume_momentum'] < 0),
        -1,  # Rising prices on declining volume (negative divergence)
        np.where(
            (data['price_momentum'] < 0) & (data['volume_momentum'] > 0),
            -1,  # Falling prices with increasing volume (negative divergence)
            1     # Healthy alignment
        )
    )
    
    # Micro-Price Volume Efficiency
    data['price_impact_per_volume'] = abs(data['price_change']) / (data['volume'] + 1e-8)
    data['vwap'] = (data['amount'] / data['volume']).replace([np.inf, -np.inf], np.nan)
    data['price_vwap_deviation'] = (data['close'] - data['vwap']) / data['close']
    
    # Volume efficiency score
    data['volume_efficiency'] = np.where(
        data['up_day'] == 1,
        data['price_change'] / (data['volume_change'] + 1e-8),
        -data['price_change'] / (data['volume_change'] + 1e-8)
    )
    data['volume_efficiency'] = data['volume_efficiency'].replace([np.inf, -np.inf], 0)
    data['volume_efficiency_smooth'] = data['volume_efficiency'].rolling(window=5, min_periods=3).mean()
    
    # Liquidity Stress Testing
    # Market Depth Proxy Analysis
    data['slippage_ratio'] = (data['high'] - data['low']) / (abs(data['close'] - data['close'].shift(1)) + 1e-8)
    data['gap_size'] = abs(data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_range_ratio'] = data['gap_size'] / (data['intraday_vol'] + 1e-8)
    
    # Volume Liquidity Assessment
    data['volume_20d_avg'] = data['volume'].rolling(window=20, min_periods=10).mean()
    data['volume_concentration'] = data['volume'] / data['volume_20d_avg']
    
    # Trade size analysis using amount/volume as proxy
    data['avg_trade_size'] = data['amount'] / (data['volume'] + 1e-8)
    data['trade_size_trend'] = data['avg_trade_size'].rolling(window=5, min_periods=3).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 3 else 0
    )
    
    # Stress Scenario Identification
    data['liquidity_drought'] = ((data['volume'] < data['volume_20d_avg'] * 0.7) & 
                                (data['volume_concentration'].rolling(window=3).sum() >= 2)).astype(int)
    
    data['liquidity_crisis'] = ((data['volume'] > data['volume_20d_avg'] * 1.5) & 
                               (data['gap_size'] > data['gap_size'].rolling(window=20).mean() * 2) &
                               (data['slippage_ratio'] > 2)).astype(int)
    
    # Regime-Adaptive Signal Integration
    # Volatility-weighted divergence signals
    base_divergence = data['price_vol_divergence'] * data['volume_efficiency_smooth']
    
    # High volatility regime adjustments
    high_vol_weight = np.where(data['vol_regime'] == 2, 1.5, 1.0)
    
    # Low volatility regime adjustments  
    low_vol_weight = np.where(data['vol_regime'] == 0, 0.7, 1.0)
    
    # Liquidity stress adjustments
    liquidity_weight = np.where(data['liquidity_drought'] == 1, 0.3, 
                               np.where(data['liquidity_crisis'] == 1, 0.1, 1.0))
    
    # Final alpha factor calculation
    alpha_factor = (base_divergence * high_vol_weight * low_vol_weight * liquidity_weight * 
                   (1 + data['trade_size_trend']))
    
    # Smooth the final factor
    alpha_smooth = alpha_factor.rolling(window=3, min_periods=2).mean()
    
    return alpha_smooth
