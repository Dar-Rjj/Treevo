import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Regime Adaptive Reversal Factor
    """
    # Calculate basic price returns
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    
    # Volatility Regime Classification
    df['intraday_vol'] = (df['high'] - df['low']) / df['close']
    df['vol_regime_threshold'] = df['intraday_vol'].rolling(window=20, min_periods=10).quantile(0.7)
    df['high_vol_regime'] = (df['intraday_vol'] > df['vol_regime_threshold']).astype(int)
    
    # Price Reversal Patterns
    df['price_change'] = (df['close'] - df['open']) / df['open']
    df['returns_std_10'] = df['returns'].rolling(window=10, min_periods=5).std()
    
    # Extreme Move Detection
    df['large_up_move'] = (df['price_change'] > df['returns_std_10']).astype(int)
    df['large_down_move'] = ((-df['price_change']) > df['returns_std_10']).astype(int)
    df['extreme_move_flag'] = (df['large_up_move'] | df['large_down_move']).astype(int)
    
    # Reversal Strength (using only historical data)
    df['return_sign'] = np.sign(df['returns'])
    df['abs_return'] = np.abs(df['returns'])
    
    # Calculate reversal persistence using only past data
    df['reversal_sign_1'] = -df['return_sign'].shift(1) * df['return_sign'].shift(2)
    df['reversal_sign_2'] = -df['return_sign'].shift(2) * df['return_sign'].shift(3)
    df['reversal_sign_3'] = -df['return_sign'].shift(3) * df['return_sign'].shift(4)
    df['reversal_persistence'] = df[['reversal_sign_1', 'reversal_sign_2', 'reversal_sign_3']].sum(axis=1, skipna=True)
    
    # Calculate reversal magnitude using only past data
    df['reversal_magnitude'] = (df['abs_return'].shift(1) / df['abs_return'].shift(2)).replace([np.inf, -np.inf], np.nan)
    
    # Volume Confirmation
    df['volume_median_10'] = df['volume'].rolling(window=10, min_periods=5).median()
    df['volume_spike'] = (df['volume'] / df['volume_median_10'] > 2).astype(int)
    
    df['volume_change'] = df['volume'].pct_change()
    df['volume_price_divergence'] = (np.sign(df['returns']) != np.sign(df['volume_change'])).astype(int)
    
    # Regime-Adaptive Signal Generation
    # High Volatility Signals
    df['quick_reversal'] = df['extreme_move_flag'] * df['reversal_magnitude'].shift(1)
    df['volume_confirmed_reversal'] = df['volume_spike'] * df['reversal_magnitude'].shift(1)
    
    # Low Volatility Signals
    df['delayed_reversal'] = df['reversal_persistence']
    df['subtle_divergence'] = df['volume_price_divergence'] * df['returns']
    
    # Mean-reversion bias using autocorrelation
    df['autocorr_1'] = df['returns'].rolling(window=10, min_periods=5).apply(
        lambda x: x.autocorr(lag=1) if len(x) >= 5 else np.nan, raw=False
    )
    df['mean_reversion_bias'] = -df['autocorr_1']
    
    # Signal Blending
    high_vol_signal = 0.6 * df['quick_reversal'] + 0.4 * df['volume_confirmed_reversal']
    low_vol_signal = 0.5 * df['delayed_reversal'] + 0.3 * df['subtle_divergence'] + 0.2 * df['mean_reversion_bias']
    
    # Regime-weighted combination
    df['regime_adaptive_signal'] = (
        high_vol_signal * df['high_vol_regime'] + 
        low_vol_signal * (1 - df['high_vol_regime'])
    )
    
    # Dynamic weighting based on recent regime persistence
    df['regime_persistence'] = df['high_vol_regime'].rolling(window=5, min_periods=3).mean()
    df['dynamic_weight'] = 0.5 + 0.3 * (2 * df['regime_persistence'] - 1)  # Adjust between 0.2 and 0.8
    
    df['final_signal'] = (
        df['dynamic_weight'] * high_vol_signal + 
        (1 - df['dynamic_weight']) * low_vol_signal
    )
    
    # Risk Adjustment
    df['returns_std_20'] = df['returns'].rolling(window=20, min_periods=10).std()
    df['volatility_scaled_signal'] = df['final_signal'] / df['returns_std_20']
    
    # Drawdown protection
    df['cumulative_max'] = df['close'].expanding().max()
    df['drawdown'] = (df['close'] - df['cumulative_max']) / df['cumulative_max']
    df['max_drawdown_ratio'] = df['drawdown'].rolling(window=20, min_periods=10).min()
    df['drawdown_adjusted_signal'] = df['volatility_scaled_signal'] * (1 - df['max_drawdown_ratio'])
    
    # Final factor with performance enhancements
    df['factor'] = df['drawdown_adjusted_signal']
    
    # Clean up and return
    result = df['factor'].copy()
    result = result.replace([np.inf, -np.inf], np.nan)
    
    return result
