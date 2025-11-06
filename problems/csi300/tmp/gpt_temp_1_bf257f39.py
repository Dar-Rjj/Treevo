import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility Regime Adaptive Momentum factor that adjusts momentum calculation
    based on detected volatility regime shifts and persistence patterns.
    """
    # Calculate basic price features
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['high_low_range'] = (df['high'] - df['low']) / df['close']
    
    # 1. Volatility Persistence Calculation
    # Rolling realized volatility (20-day)
    df['realized_vol'] = df['returns'].rolling(window=20).std()
    
    # Volatility autocorrelation (lag 1)
    df['vol_autocorr'] = df['realized_vol'].rolling(window=30).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else np.nan, raw=False
    )
    
    # 2. Volatility Regime Detection
    # Volatility breakout detection (2 standard deviation move)
    vol_ma = df['realized_vol'].rolling(window=30).mean()
    vol_std = df['realized_vol'].rolling(window=30).std()
    df['vol_breakout'] = (df['realized_vol'] > (vol_ma + 2 * vol_std)).astype(int)
    
    # Mean-reversion in volatility flag
    df['vol_mean_reversion'] = ((df['realized_vol'] < vol_ma) & 
                               (df['realized_vol'].shift(1) > vol_ma)).astype(int)
    
    # Volatility regime classification
    high_vol_threshold = df['realized_vol'].rolling(window=60).quantile(0.7)
    low_vol_threshold = df['realized_vol'].rolling(window=60).quantile(0.3)
    
    df['vol_regime'] = 0  # Normal regime
    df.loc[df['realized_vol'] > high_vol_threshold, 'vol_regime'] = 1  # High volatility
    df.loc[df['realized_vol'] < low_vol_threshold, 'vol_regime'] = -1  # Low volatility
    
    # 3. Regime Duration Strength
    regime_changes = (df['vol_regime'] != df['vol_regime'].shift(1)).cumsum()
    df['regime_duration'] = regime_changes.groupby(regime_changes).cumcount() + 1
    
    # 4. Adaptive Momentum Calculation
    # Short lookback momentum for high volatility (5 days)
    df['momentum_short'] = df['close'] / df['close'].shift(5) - 1
    
    # Long lookback momentum for low volatility (20 days)
    df['momentum_long'] = df['close'] / df['close'].shift(20) - 1
    
    # Volatility-normalized returns
    df['vol_normalized_returns'] = df['returns'] / (df['realized_vol'] + 1e-8)
    df['vol_normalized_momentum'] = df['vol_normalized_returns'].rolling(window=10).mean()
    
    # 5. Price breakouts for low volatility regime
    price_range = df['high'].rolling(window=20).max() - df['low'].rolling(window=20).min()
    df['price_breakout'] = (df['close'] - df['close'].shift(1)) / (price_range + 1e-8)
    
    # 6. Generate regime-aware signals
    # High volatility regime: trend following with volatility normalization
    high_vol_signal = df['vol_normalized_momentum'] * np.sqrt(df['regime_duration'])
    
    # Low volatility regime: focus on price breakouts with longer momentum
    low_vol_signal = df['momentum_long'] * df['price_breakout'] * np.sqrt(df['regime_duration'])
    
    # Normal regime: blend of strategies
    normal_signal = 0.6 * df['momentum_short'] + 0.4 * df['momentum_long']
    
    # 7. Combine signals based on volatility regime
    factor = np.where(
        df['vol_regime'] == 1, 
        high_vol_signal,
        np.where(
            df['vol_regime'] == -1,
            low_vol_signal,
            normal_signal
        )
    )
    
    # 8. Apply regime confidence weighting
    vol_persistence_strength = df['vol_autocorr'].abs()
    regime_confidence = np.minimum(df['regime_duration'] / 10, 1)  # Cap at 10 days
    
    # Final factor with confidence adjustment
    final_factor = factor * regime_confidence * (1 + vol_persistence_strength)
    
    # Clean and return
    result = pd.Series(final_factor, index=df.index)
    result = result.replace([np.inf, -np.inf], np.nan)
    
    return result
