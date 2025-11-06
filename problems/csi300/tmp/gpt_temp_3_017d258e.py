import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate novel alpha factor combining volatility fractality, order flow imbalance, 
    and momentum efficiency concepts
    """
    # Create copy to avoid modifying original dataframe
    data = df.copy()
    
    # 1. Volatility Fractality Analysis
    # Intraday Volatility Ratio with stability check
    intraday_vol_ratio = (data['high'] - data['low']) / (np.abs(data['close'] - data['open']) + 1e-8)
    
    # Multi-scale volatility clustering using different rolling windows
    vol_cluster_5d = data['high'].rolling(window=5).std() / (data['low'].rolling(window=5).mean() + 1e-8)
    vol_cluster_10d = data['high'].rolling(window=10).std() / (data['low'].rolling(window=10).mean() + 1e-8)
    vol_clustering_pattern = vol_cluster_5d / (vol_cluster_10d + 1e-8)
    
    # Fractal breakdown in price-volume correlation
    price_volume_corr_5d = data['close'].rolling(window=5).corr(data['volume'])
    price_volume_corr_10d = data['close'].rolling(window=10).corr(data['volume'])
    fractal_corr_breakdown = price_volume_corr_5d - price_volume_corr_10d
    
    # 2. Order Flow Imbalance Detection
    # Buy-Sell Pressure Differential using amount and volume
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    money_flow = typical_price * data['volume']
    money_flow_5d = money_flow.rolling(window=5).mean()
    money_flow_10d = money_flow.rolling(window=10).mean()
    flow_pressure = money_flow_5d / (money_flow_10d + 1e-8)
    
    # Multi-timeframe flow momentum
    flow_momentum_5d = money_flow.diff(5) / (money_flow.shift(5) + 1e-8)
    flow_momentum_10d = money_flow.diff(10) / (money_flow.shift(10) + 1e-8)
    multi_timeframe_momentum = flow_momentum_5d - flow_momentum_10d
    
    # Flow asymmetry reversal patterns
    high_low_range = data['high'] - data['low']
    open_close_range = np.abs(data['close'] - data['open'])
    flow_asymmetry = (high_low_range - open_close_range) / (high_low_range + 1e-8)
    flow_asymmetry_reversal = flow_asymmetry - flow_asymmetry.rolling(window=5).mean()
    
    # 3. Momentum Efficiency Scoring
    # Momentum Efficiency: Price Change / Volume Effort
    price_change_5d = data['close'].pct_change(5)
    volume_effort_5d = data['volume'].rolling(window=5).sum() / (data['volume'].rolling(window=20).sum() + 1e-8)
    momentum_efficiency = price_change_5d / (volume_effort_5d + 1e-8)
    
    # Volume-Price Synchronization Quality
    price_trend_5d = data['close'].rolling(window=5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    volume_trend_5d = data['volume'].rolling(window=5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    sync_quality = np.sign(price_trend_5d * volume_trend_5d) * np.abs(price_trend_5d)
    
    # Momentum Degradation Detection
    momentum_5d = data['close'].pct_change(5)
    momentum_10d = data['close'].pct_change(10)
    momentum_degradation = momentum_5d - momentum_10d
    
    # Combine all components into final factor
    # Normalize each component by its rolling standard deviation
    def normalize_series(series, window=20):
        return (series - series.rolling(window=window).mean()) / (series.rolling(window=window).std() + 1e-8)
    
    # Weighted combination of normalized components
    factor = (
        0.15 * normalize_series(intraday_vol_ratio) +
        0.12 * normalize_series(vol_clustering_pattern) +
        0.10 * normalize_series(fractal_corr_breakdown) +
        0.15 * normalize_series(flow_pressure) +
        0.13 * normalize_series(multi_timeframe_momentum) +
        0.10 * normalize_series(flow_asymmetry_reversal) +
        0.15 * normalize_series(momentum_efficiency) +
        0.05 * normalize_series(sync_quality) +
        0.05 * normalize_series(momentum_degradation)
    )
    
    return factor
