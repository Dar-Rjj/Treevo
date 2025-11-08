import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Price-Volume Asymmetry with Regime Detection alpha factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate returns for correlation calculations
    data['returns'] = data['close'].pct_change()
    
    # 1. Calculate Asymmetric Price-Volume Relationships
    # Short-term (3-day) price-volume correlation
    data['short_term_corr'] = data['returns'].rolling(window=3).corr(data['volume'].pct_change())
    
    # Medium-term (10-day) price-volume correlation
    data['medium_term_corr'] = data['returns'].rolling(window=10).corr(data['volume'].pct_change())
    
    # Correlation asymmetry
    data['corr_asymmetry'] = data['short_term_corr'] - data['medium_term_corr']
    
    # Identify correlation regime shifts (significant changes in asymmetry)
    data['corr_regime_shift'] = data['corr_asymmetry'].diff().abs() > data['corr_asymmetry'].rolling(window=20).std()
    
    # 2. Detect Market Microstructure Regimes
    # Bid-ask spread proxy using daily high-low range normalized by close
    data['spread_proxy'] = (data['high'] - data['low']) / data['close']
    
    # Volume concentration (top 30% volume days / total volume over 10 days)
    def volume_concentration(volumes):
        if len(volumes) < 10:
            return np.nan
        sorted_vol = np.sort(volumes)
        threshold = sorted_vol[int(0.7 * len(sorted_vol))]
        top_volume = volumes[volumes >= threshold].sum()
        return top_volume / volumes.sum() if volumes.sum() > 0 else 0
    
    data['volume_concentration'] = data['volume'].rolling(window=10).apply(volume_concentration, raw=True)
    
    # Price efficiency using 5-day return autocorrelation
    data['return_autocorr'] = data['returns'].rolling(window=5).apply(lambda x: pd.Series(x).autocorr(), raw=False)
    
    # Classify regimes based on microstructure characteristics
    data['high_liquidity_regime'] = (data['spread_proxy'] < data['spread_proxy'].rolling(window=20).quantile(0.3)) & \
                                   (data['volume_concentration'] < 0.5)
    
    data['low_liquidity_regime'] = (data['spread_proxy'] > data['spread_proxy'].rolling(window=20).quantile(0.7)) & \
                                  (data['volume_concentration'] > 0.7)
    
    data['efficient_regime'] = data['return_autocorr'].abs() < 0.1
    
    # 3. Generate Regime-Adaptive Alpha Signal
    # Base signal from correlation asymmetry
    base_signal = data['corr_asymmetry'].fillna(0)
    
    # Regime-specific signal amplification
    signal_amplification = np.ones(len(data))
    
    # Strong positive amplification in high liquidity + efficient regimes
    high_liquidity_mask = data['high_liquidity_regime'].fillna(False) & data['efficient_regime'].fillna(False)
    signal_amplification[high_liquidity_mask] = 1.5
    
    # Negative amplification in low liquidity regimes with correlation shifts
    low_liquidity_shift = data['low_liquidity_regime'].fillna(False) & data['corr_regime_shift'].fillna(False)
    signal_amplification[low_liquidity_shift] = -1.2
    
    # Apply regime-specific adjustments
    regime_adjusted_signal = base_signal * signal_amplification
    
    # Calculate signal persistence (3-day rolling mean to smooth)
    final_signal = regime_adjusted_signal.rolling(window=3, min_periods=1).mean()
    
    # Normalize the final signal
    final_signal = (final_signal - final_signal.rolling(window=20).mean()) / final_signal.rolling(window=20).std()
    
    return final_signal
