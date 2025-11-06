import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Volatility Regime Momentum Factor
    # Realized volatility (20-day rolling std of close)
    realized_vol = df['close'].rolling(window=20).std()
    
    # Range volatility (10-day average of high-low range)
    daily_range = df['high'] - df['low']
    range_vol = daily_range.rolling(window=10).mean()
    
    # Volatility ratio
    vol_ratio = realized_vol / range_vol
    
    # Volatility acceleration (5-day change in volatility ratio)
    vol_acceleration = vol_ratio - vol_ratio.shift(5)
    
    # Regime break (volatility ratio crosses 80th percentile over 20 days)
    vol_ratio_80q = vol_ratio.rolling(window=20).quantile(0.8)
    regime_break = (vol_ratio > vol_ratio_80q).astype(int)
    
    # Price momentum components
    short_momentum = df['close'] / df['close'].shift(3) - 1
    medium_momentum = df['close'] / df['close'].shift(8) - 1
    
    # Momentum consistency (correlation of recent price changes)
    momentum_consistency = df['close'].rolling(window=5).apply(
        lambda x: np.corrcoef(range(5), x)[0,1] if len(x) == 5 else np.nan, 
        raw=True
    )
    
    # Volume confirmation
    vol_ma_4 = df['volume'].rolling(window=4).mean().shift(1)
    volume_momentum = df['volume'] / vol_ma_4
    volume_vol_ratio = df['volume'] / (df['high'] - df['low'])
    volume_persistence = df['volume'] / df['volume'].shift(1)
    
    # Volatility regime momentum factor
    regime_momentum = vol_acceleration * momentum_consistency
    volume_adjustment = volume_momentum * volume_vol_ratio
    volatility_factor = regime_momentum * volume_adjustment * regime_break * short_momentum
    
    # Price Elasticity Factor
    # Elasticity measurement
    return_magnitude = abs(df['close'] / df['close'].shift(1) - 1)
    range_utilization = (df['high'] - df['low']) / (df['high'].shift(1) - df['low'].shift(1))
    elasticity_score = return_magnitude / range_utilization
    
    # Volume sensitivity
    volume_elasticity = (df['volume'] / df['volume'].shift(1)) / abs(df['close'] / df['close'].shift(1) - 1)
    volume_clustering = df['volume'] / df['volume'].rolling(window=4).mean().shift(1)
    sensitivity_trend = volume_elasticity - volume_elasticity.shift(3)
    
    # Market memory
    # Price memory (auto-correlation)
    def price_autocorr(x):
        if len(x) < 5:
            return np.nan
        return np.corrcoef(x[:-1], x[1:])[0,1]
    
    price_auto_corr = df['close'].rolling(window=5).apply(price_autocorr, raw=True)
    memory_decay = 1 - abs(price_auto_corr)
    
    # Volume memory
    volume_persistence_elasticity = df['volume'] / df['volume'].shift(1)
    
    def volume_autocorr(x):
        if len(x) < 5:
            return np.nan
        return np.corrcoef(x[:-1], x[1:])[0,1]
    
    memory_strength = df['volume'].rolling(window=5).apply(volume_autocorr, raw=True)
    
    # Price elasticity factor
    elasticity_momentum = elasticity_score * sensitivity_trend
    memory_adjustment = price_auto_corr * volume_persistence_elasticity
    elasticity_factor = elasticity_momentum * memory_decay * volume_clustering
    
    # Microstructure Momentum Factor
    # Intraday dynamics
    opening_gap = df['open'] / df['close'].shift(1) - 1
    
    # Since we don't have intraday data, we'll use proxies
    # Opening volume intensity proxy: first 30min volume / total volume (using daily patterns)
    opening_volume_intensity = df['volume'] / df['volume'].rolling(window=5).mean()
    
    opening_momentum = (df['close'] - df['open']) / (df['high'] - df['low'])
    
    # Closing behavior
    closing_pressure = (df['close'] - df['low']) / (df['high'] - df['low'])
    
    # Final hour volume proxy
    final_hour_volume = df['volume'] / df['volume'].rolling(window=5).mean()
    
    closing_efficiency = abs(df['close'] - df['close'].shift(1)) / (df['high'] - df['low'])
    
    # Order flow analysis
    # Buy/sell pressure proxies
    price_change = df['close'] - df['close'].shift(1)
    buy_pressure = (price_change > 0).astype(int) * df['volume'] / df['volume']
    sell_pressure = (price_change < 0).astype(int) * df['volume'] / df['volume']
    flow_momentum = buy_pressure - buy_pressure.shift(1)
    
    # Flow concentration proxies
    volume_skewness = df['volume'].rolling(window=10).apply(
        lambda x: (np.percentile(x, 75) - np.median(x)) / (np.median(x) - np.percentile(x, 25)) if len(x) == 10 else np.nan,
        raw=True
    )
    
    def volume_autocorr_micro(x):
        if len(x) < 5:
            return np.nan
        return np.corrcoef(x[:-1], x[1:])[0,1]
    
    flow_persistence = df['volume'].rolling(window=5).apply(volume_autocorr_micro, raw=True)
    
    # Concentration ratio proxy
    concentration_ratio = df['volume'] / df['volume'].rolling(window=5).mean()
    
    # Microstructure momentum factor
    micro_momentum = opening_momentum * closing_pressure * flow_momentum
    flow_quality = flow_persistence * concentration_ratio
    microstructure_factor = micro_momentum * flow_quality * opening_volume_intensity
    
    # Combine all factors with equal weighting
    combined_factor = (
        volatility_factor.fillna(0) + 
        elasticity_factor.fillna(0) + 
        microstructure_factor.fillna(0)
    ) / 3
    
    return combined_factor
