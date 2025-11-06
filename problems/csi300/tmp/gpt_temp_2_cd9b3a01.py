import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor using multiple heuristics including:
    - Volume-Price Fractal Divergence
    - Regime-Adaptive Pressure Imbalance
    - Efficiency-Weighted Breakout
    - Volume Cluster Acceleration
    - Multi-Timeframe Mean Reversion
    - Volatility-Volume Momentum Confluence
    """
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Required columns
    cols_required = ['open', 'high', 'low', 'close', 'volume', 'amount']
    if not all(col in df.columns for col in cols_required):
        return result
    
    # 1. Volume-Price Fractal Divergence
    def hurst_exponent(series, window):
        """Approximate Hurst exponent for fractality measurement"""
        hurst_values = pd.Series(index=series.index, dtype=float)
        for i in range(window, len(series)):
            window_data = series.iloc[i-window:i]
            if len(window_data) < window:
                continue
            lags = range(2, min(10, len(window_data)))
            tau = [np.std(np.subtract(window_data[lag:].values, window_data[:-lag].values)) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            hurst_values.iloc[i] = poly[0]
        return hurst_values
    
    price_hurst = hurst_exponent(df['close'], 10)
    volume_hurst = hurst_exponent(df['volume'], 5)
    fractal_divergence = (volume_hurst - price_hurst).fillna(0)
    
    # 2. Regime-Adaptive Pressure Imbalance
    def atr(high, low, close, window):
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window).mean()
    
    volatility_regime = atr(df['high'], df['low'], df['close'], 10)
    trend_regime = (df['close'] / df['close'].rolling(20).mean() - 1).fillna(0)
    
    # Intraday momentum components
    upward_pressure = ((df['close'] - df['open']) / df['open']).rolling(3).mean().fillna(0)
    downward_pressure = ((df['open'] - df['low']) / df['open']).rolling(3).mean().fillna(0)
    
    # Adaptive pressure signal
    pressure_ratio = upward_pressure / (downward_pressure + 1e-8)
    regime_adaptive_pressure = pressure_ratio * (1 - abs(trend_regime)) + pressure_ratio * abs(trend_regime)
    
    # 3. Efficiency-Weighted Breakout
    def autocorr(series, window, lag=1):
        """Rolling autocorrelation"""
        autocorr_values = pd.Series(index=series.index, dtype=float)
        for i in range(window, len(series)):
            window_data = series.iloc[i-window:i]
            if len(window_data) < window:
                continue
            autocorr_values.iloc[i] = window_data.autocorr(lag=lag)
        return autocorr_values
    
    returns = df['close'].pct_change().fillna(0)
    market_returns = returns  # Using same stock as market proxy
    
    efficiency_autocorr = autocorr(returns, 10, 1)
    efficiency_delay = pd.Series(index=df.index, dtype=float)
    for i in range(10, len(df)):
        window_returns = returns.iloc[i-10:i]
        window_market_returns = market_returns.iloc[i-10:i-1]  # Lagged market returns
        if len(window_returns) == 10 and len(window_market_returns) == 9:
            efficiency_delay.iloc[i] = np.corrcoef(window_returns.iloc[1:], window_market_returns)[0,1]
    
    market_efficiency = (efficiency_autocorr + efficiency_delay.fillna(0)) / 2
    
    # Breakout components
    resistance = df['high'].rolling(10).max().shift(1)
    support = df['low'].rolling(10).min().shift(1)
    upside_potential = (df['high'] - resistance) / resistance
    downside_potential = (support - df['low']) / support
    
    # Weighted breakout signal
    breakout_signal = (upside_potential - downside_potential) * (1 - market_efficiency.fillna(0))
    
    # 4. Volume Cluster Acceleration
    volume_ma_20 = df['volume'].rolling(20).mean()
    abnormal_volume = (df['volume'] / volume_ma_20 > 2).astype(int)
    volume_persistence = abnormal_volume.rolling(3).sum()
    
    def momentum(series, period):
        return series - series.shift(period)
    
    price_acceleration = momentum(momentum(df['close'], 5), 3)
    volume_acceleration = momentum(momentum(df['volume'], 5), 3)
    
    volume_cluster_signal = (price_acceleration * volume_acceleration * volume_persistence).fillna(0)
    
    # 5. Multi-Timeframe Mean Reversion
    short_term_reversion = (df['close'] / df['close'].rolling(5).mean() - 1).fillna(0)
    medium_term_reversion = (df['close'] / df['close'].rolling(10).mean() - 1).fillna(0)
    
    short_term_momentum = momentum(df['close'], 3)
    medium_term_momentum = momentum(df['close'], 8)
    
    mean_reversion_signal = -(short_term_reversion * short_term_momentum + medium_term_reversion * medium_term_momentum)
    
    # 6. Volatility-Volume Momentum Confluence
    volatility_range = (df['high'] - df['low']).rolling(8).mean()
    
    # Volume-volatility correlation
    vol_vol_corr = pd.Series(index=df.index, dtype=float)
    for i in range(6, len(df)):
        vol_window = df['volume'].iloc[i-6:i]
        vol_range_window = (df['high'] - df['low']).iloc[i-6:i]
        if len(vol_window) == 6:
            vol_vol_corr.iloc[i] = np.corrcoef(vol_window, vol_range_window)[0,1]
    
    opening_momentum = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    intraday_persistence = ((df['high'] - df['open']) / df['open'] + (df['open'] - df['low']) / df['open']) / 2
    
    confluence_signal = (opening_momentum * intraday_persistence * vol_vol_corr.fillna(0) * volatility_range).fillna(0)
    
    # Combine all signals with equal weights
    combined_signal = (
        fractal_divergence.fillna(0) +
        regime_adaptive_pressure.fillna(0) +
        breakout_signal.fillna(0) +
        volume_cluster_signal.fillna(0) +
        mean_reversion_signal.fillna(0) +
        confluence_signal.fillna(0)
    ) / 6
    
    # Normalize the final signal
    result = (combined_signal - combined_signal.rolling(20).mean()) / (combined_signal.rolling(20).std() + 1e-8)
    
    return result.fillna(0)
