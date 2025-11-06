import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Intraday Momentum Divergence
    intraday_change = data['high'] - data['low']
    gap = data['close'] - data['open']
    
    # Normalize both series
    intraday_norm = intraday_change / intraday_change.rolling(window=20, min_periods=10).std()
    gap_norm = gap / gap.rolling(window=20, min_periods=10).std()
    
    # Calculate divergence based on sign comparison and magnitude difference
    sign_divergence = np.where(
        np.sign(intraday_change) != np.sign(gap), 
        np.abs(intraday_norm - gap_norm), 
        0
    )
    
    # Volume confirmation
    volume_percentile = data['volume'].rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0
    )
    volume_strength = np.tanh(volume_percentile)
    
    intraday_factor = sign_divergence * volume_strength
    
    # High-Low Range Compression Breakout
    prev_close = data['close'].shift(1)
    true_range = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            np.abs(data['high'] - prev_close),
            np.abs(data['low'] - prev_close)
        )
    )
    
    compression_ratio = true_range / true_range.rolling(window=20, min_periods=10).mean()
    
    # Volume breakout detection
    volume_median = data['volume'].rolling(window=20, min_periods=10).median()
    volume_breakout = np.where(data['volume'] > volume_median, 1.5, 0.5)
    
    # Price direction for sign
    price_trend = np.sign(data['close'] - data['close'].rolling(window=5, min_periods=3).mean())
    
    compression_factor = compression_ratio * volume_breakout * price_trend
    
    # Volatility-Regressed Return Momentum
    returns_5d = data['close'].pct_change(5)
    volatility = ((data['high'] - data['low']) / data['close']).rolling(window=20, min_periods=10).mean()
    
    # Rolling regression beta
    def calc_beta(window_returns, window_vol):
        if len(window_returns) < 10 or window_vol.std() == 0:
            return 1.0
        cov = window_returns.cov(window_vol)
        var = window_vol.var()
        return cov / var if var > 0 else 1.0
    
    beta = pd.Series(index=data.index, dtype=float)
    for i in range(20, len(data)):
        if i >= 20:
            window_returns = returns_5d.iloc[i-19:i+1]
            window_vol = volatility.iloc[i-19:i+1]
            beta.iloc[i] = calc_beta(window_returns, window_vol)
    
    # Fill early values
    beta = beta.fillna(1.0)
    
    # Generate signal with volume weighting
    volatility_factor = returns_5d * (1 / np.maximum(np.abs(beta), 0.1)) * np.tanh(data['volume'] / data['volume'].rolling(window=20, min_periods=10).mean())
    
    # Price-Volume Trend Convergence
    # Price EMAs
    ema_5 = data['close'].ewm(span=5, adjust=False).mean()
    ema_10 = data['close'].ewm(span=10, adjust=False).mean()
    ema_20 = data['close'].ewm(span=20, adjust=False).mean()
    
    # Price convergence score (normalized difference between EMAs)
    price_conv = (np.abs(ema_5 - ema_10) + np.abs(ema_10 - ema_20) + np.abs(ema_5 - ema_20)) / 3
    price_conv_score = 1 - (price_conv / price_conv.rolling(window=20, min_periods=10).std())
    
    # Volume EMAs
    vol_ema_5 = data['volume'].ewm(span=5, adjust=False).mean()
    vol_ema_10 = data['volume'].ewm(span=10, adjust=False).mean()
    vol_ema_20 = data['volume'].ewm(span=20, adjust=False).mean()
    
    # Volume convergence score
    vol_conv = (np.abs(vol_ema_5 - vol_ema_10) + np.abs(vol_ema_10 - vol_ema_20) + np.abs(vol_ema_5 - vol_ema_20)) / 3
    vol_conv_score = 1 - (vol_conv / vol_conv.rolling(window=20, min_periods=10).std())
    
    # Synchronization (correlation between price and volume trends)
    price_trend_dir = np.sign(ema_5 - ema_20)
    vol_trend_dir = np.sign(vol_ema_5 - vol_ema_20)
    sync_multiplier = np.where(price_trend_dir == vol_trend_dir, 1.5, 0.5)
    
    trend_factor = price_conv_score * vol_conv_score * sync_multiplier
    
    # Overnight Gap Mean Reversion Strength
    overnight_gaps = (data['open'] - prev_close) / prev_close
    
    # Gap persistence (consecutive gaps in same direction)
    gap_persistence = pd.Series(0, index=data.index)
    for i in range(1, len(data)):
        if i >= 2:
            if np.sign(overnight_gaps.iloc[i]) == np.sign(overnight_gaps.iloc[i-1]):
                gap_persistence.iloc[i] = gap_persistence.iloc[i-1] + 1
    
    persistence_indicator = np.tanh(gap_persistence / 5)  # Scale persistence
    
    # Volume participation
    vol_avg = data['volume'].rolling(window=10, min_periods=5).mean()
    volume_ratio = data['volume'] / vol_avg
    volume_participation = np.tanh(volume_ratio - 1)
    
    # Mean reversion signal (contrarian)
    gap_factor = -overnight_gaps * persistence_indicator * volume_participation
    
    # Combine all factors with equal weighting
    combined_factor = (
        intraday_factor.fillna(0) + 
        compression_factor.fillna(0) + 
        volatility_factor.fillna(0) + 
        trend_factor.fillna(0) + 
        gap_factor.fillna(0)
    ) / 5
    
    return combined_factor
