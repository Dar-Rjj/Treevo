import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Volatility Momentum with Fractal Volume Confirmation
    """
    data = df.copy()
    
    # Multi-Scale Volatility Regime Analysis
    # Fractal Volatility Detection
    def calculate_atr(high, low, close, window):
        tr = np.maximum(high - low, 
                       np.maximum(abs(high - close.shift(1)), 
                                 abs(low - close.shift(1))))
        return tr.rolling(window=window).mean()
    
    atr_5d = calculate_atr(data['high'], data['low'], data['close'], 5)
    atr_20d = calculate_atr(data['high'], data['low'], data['close'], 20)
    volatility_ratio = atr_5d / atr_20d
    volatility_breakout = volatility_ratio - 1
    
    # Volume Regime Classification - Hurst exponent approximation
    def hurst_approximation(series, window=20):
        lags = range(2, min(6, window))
        tau = []
        for lag in lags:
            ts = series.diff(lag).dropna()
            if len(ts) > 0:
                tau.append(np.sqrt(np.mean(ts**2)))
            else:
                tau.append(1.0)
        
        if len(tau) > 1:
            x = np.log(lags[:len(tau)])
            y = np.log(tau)
            if len(x) > 1 and not np.any(np.isinf(y)) and not np.any(np.isnan(y)):
                return np.polyfit(x, y, 1)[0]
        return 0.5
    
    volume_hurst = data['volume'].rolling(window=20).apply(
        lambda x: hurst_approximation(x), raw=False
    )
    volume_regime = (volume_hurst > 0.6).astype(int)  # Trending if > 0.6
    
    # Intraday Momentum Fractal Analysis
    high_to_close = (data['high'] - data['close']) / data['close']
    high_to_open = (data['high'] - data['open']) / data['open']
    low_to_close = (data['close'] - data['low']) / data['close']
    low_to_open = (data['open'] - data['low']) / data['open']
    
    upside_acceleration = high_to_close - high_to_open
    downside_acceleration = low_to_close - low_to_open
    momentum_divergence = upside_acceleration - downside_acceleration
    
    # Volume-Price Fractal Integration
    # Smart Volume Confirmation
    volume_5d_avg = data['volume'].rolling(window=5).mean()
    volume_intensity = data['volume'] / volume_5d_avg
    
    volume_3d_avg = data['volume'].rolling(window=3).mean()
    volume_10d_avg = data['volume'].rolling(window=10).mean()
    volume_trend_ratio = volume_3d_avg / volume_10d_avg
    
    # Volume Cluster Detection
    def volume_cluster_metric(volume_series, window=10):
        volume_changes = volume_series.pct_change().dropna()
        if len(volume_changes) >= window:
            recent_vol = volume_changes.iloc[-window:]
            cluster_strength = np.sum(np.abs(recent_vol) > np.std(volume_changes))
            persistence = len([i for i in range(1, len(recent_vol)) 
                             if np.sign(recent_vol.iloc[i]) == np.sign(recent_vol.iloc[i-1])])
            return cluster_strength * persistence / window
        return 0
    
    volume_cluster = data['volume'].rolling(window=10).apply(
        lambda x: volume_cluster_metric(x), raw=False
    )
    
    # Bid-Ask Pressure Analysis
    daily_range = data['high'] - data['low']
    buying_pressure = np.where(daily_range > 0, 
                              (data['close'] - data['low']) / daily_range, 0)
    selling_pressure = np.where(daily_range > 0, 
                               (data['high'] - data['close']) / daily_range, 0)
    bid_ask_pressure = buying_pressure - selling_pressure
    
    # Regime-Adaptive Signal Generation
    high_vol_regime = (volatility_ratio > 1.2).astype(int)
    low_vol_regime = (volatility_ratio < 0.8).astype(int)
    transition_regime = ((volatility_ratio >= 0.8) & (volatility_ratio <= 1.2)).astype(int)
    
    # High Volatility Component
    high_vol_component = (momentum_divergence * volume_intensity * volatility_breakout)
    
    # Low Volatility Component  
    low_vol_component = (bid_ask_pressure * volume_cluster * volume_hurst)
    
    # Regime Transition Component (weighted average)
    transition_component = 0.5 * high_vol_component + 0.5 * low_vol_component
    
    # Primary Regime-Weighted Component
    regime_weighted = (
        high_vol_regime * high_vol_component +
        low_vol_regime * low_vol_component +
        transition_regime * transition_component
    )
    
    # Multi-Scale Volume Confirmation
    volume_confirmation = volume_trend_ratio * volume_hurst * volume_intensity
    
    # Final Factor with Range Adjustment
    final_factor = regime_weighted * volume_confirmation
    range_adjusted_factor = np.where(daily_range > 0, final_factor / daily_range, final_factor)
    
    # Normalize and return
    factor_series = pd.Series(range_adjusted_factor, index=data.index)
    factor_series = (factor_series - factor_series.rolling(window=20).mean()) / factor_series.rolling(window=20).std()
    
    return factor_series.fillna(0)
