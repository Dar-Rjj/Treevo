import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate daily returns
    returns = df['close'].pct_change()
    
    # Volatility Classification
    vol_window = 20
    volatility = returns.rolling(window=vol_window, min_periods=10).std()
    
    # Identify volatility regimes using rolling percentiles
    vol_regime = pd.Series(np.where(
        volatility > volatility.rolling(window=60, min_periods=30).quantile(0.7),
        'high',
        np.where(
            volatility < volatility.rolling(window=60, min_periods=30).quantile(0.3),
            'low',
            'normal'
        )
    ), index=df.index)
    
    # Price Efficiency Metrics
    # Intraday range utilization
    intraday_range = df['high'] - df['low']
    price_change = np.abs(df['close'] - df['open'])
    range_utilization = intraday_range / (price_change + 1e-8)
    
    # Volume-price coordination
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).rolling(window=5, min_periods=3).sum() / df['volume'].rolling(window=5, min_periods=3).sum()
    volume_price_coord = (df['close'] - vwap) / (df['high'] - df['low'] + 1e-8)
    
    # Return distribution characteristics
    return_skew = returns.rolling(window=10, min_periods=5).skew()
    return_autocorr = returns.rolling(window=10, min_periods=5).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else 0
    )
    
    # Regime-Specific Alpha Signals
    factor_values = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i < max(vol_window, 10):
            factor_values.iloc[i] = 0
            continue
            
        current_regime = vol_regime.iloc[i]
        
        if current_regime == 'high':
            # High volatility: Extreme move reversion and momentum persistence
            recent_returns = returns.iloc[max(0, i-5):i+1]
            extreme_moves = np.abs(recent_returns) > volatility.iloc[i] * 1.5
            reversion_signal = -np.sign(returns.iloc[i]) if extreme_moves.any() else 0
            
            # Momentum persistence (trend following)
            short_ma = df['close'].iloc[max(0, i-3):i+1].mean()
            long_ma = df['close'].iloc[max(0, i-10):i+1].mean()
            momentum_signal = 1 if short_ma > long_ma else -1
            
            regime_factor = 0.6 * reversion_signal + 0.4 * momentum_signal
            
        elif current_regime == 'low':
            # Low volatility: Range breakout and volume accumulation
            recent_high = df['high'].iloc[max(0, i-5):i+1].max()
            recent_low = df['low'].iloc[max(0, i-5):i+1].min()
            
            # Range breakout signal
            if df['close'].iloc[i] > recent_high:
                breakout_signal = 1
            elif df['close'].iloc[i] < recent_low:
                breakout_signal = -1
            else:
                breakout_signal = 0
            
            # Volume accumulation
            volume_ma = df['volume'].rolling(window=10, min_periods=5).mean()
            volume_surge = df['volume'].iloc[i] > volume_ma.iloc[i] * 1.2
            volume_signal = 1 if volume_surge else 0
            
            regime_factor = 0.7 * breakout_signal + 0.3 * volume_signal
            
        else:  # normal regime
            # Combine efficiency metrics for normal volatility
            efficiency_score = (
                0.3 * (1 / (range_utilization.iloc[i] + 1e-8)) +
                0.4 * volume_price_coord.iloc[i] +
                0.2 * (-return_skew.iloc[i]) +  # Prefer negative skew (less tail risk)
                0.1 * (-return_autocorr.iloc[i])  # Prefer negative autocorrelation (mean reversion)
            )
            regime_factor = efficiency_score
        
        factor_values.iloc[i] = regime_factor
    
    # Normalize the factor
    factor_values = (factor_values - factor_values.rolling(window=60, min_periods=30).mean()) / (factor_values.rolling(window=60, min_periods=30).std() + 1e-8)
    
    return factor_values
