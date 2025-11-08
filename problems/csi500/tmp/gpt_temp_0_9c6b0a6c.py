import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Convergence Factor that combines signals from short-term (1-3 days),
    medium-term (5-10 days), and long-term (20+ days) timeframes with dynamic weighting
    based on volatility regimes.
    """
    # Calculate returns for various timeframes
    df = df.copy()
    df['returns_1d'] = df['close'].pct_change()
    df['returns_3d'] = df['close'].pct_change(3)
    df['returns_5d'] = df['close'].pct_change(5)
    df['returns_10d'] = df['close'].pct_change(10)
    df['returns_20d'] = df['close'].pct_change(20)
    
    # Short-term dynamics (1-3 days)
    # Intraday momentum persistence
    df['intraday_return'] = (df['close'] - df['open']) / df['open']
    df['intraday_momentum_streak'] = 0
    for i in range(1, len(df)):
        if (df['intraday_return'].iloc[i] * df['intraday_return'].iloc[i-1]) > 0:
            df.loc[df.index[i], 'intraday_momentum_streak'] = df['intraday_momentum_streak'].iloc[i-1] + 1
    
    # Volume confirmation patterns
    df['volume_ma_5'] = df['volume'].rolling(5).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma_5']
    df['volume_trend'] = df['volume_ratio'].rolling(3).mean()
    
    # Order flow intensity
    df['amount_ma_5'] = df['amount'].rolling(5).mean()
    df['flow_intensity'] = (df['amount'] / df['amount_ma_5']) * np.sign(df['intraday_return'])
    
    # Short-term signal
    df['short_term_signal'] = (
        df['intraday_momentum_streak'] * 0.3 +
        df['volume_trend'] * 0.4 +
        df['flow_intensity'] * 0.3
    )
    
    # Medium-term trends (5-10 days)
    # Price trend consistency
    df['ma_5'] = df['close'].rolling(5).mean()
    df['ma_10'] = df['close'].rolling(10).mean()
    df['price_trend_score'] = (
        (df['close'] > df['ma_5']).astype(int) +
        (df['close'] > df['ma_10']).astype(int) +
        (df['ma_5'] > df['ma_10']).astype(int)
    ) / 3.0
    
    # Volume trend alignment
    df['volume_ma_10'] = df['volume'].rolling(10).mean()
    df['volume_trend_ma'] = df['volume'] / df['volume_ma_10']
    df['volume_price_alignment'] = df['volume_trend_ma'] * np.sign(df['returns_5d'])
    
    # Volatility regime stability
    df['volatility_5d'] = df['returns_1d'].rolling(5).std()
    df['volatility_10d'] = df['returns_1d'].rolling(10).std()
    df['volatility_stability'] = 1 - (df['volatility_5d'] / df['volatility_10d']).abs()
    
    # Medium-term signal
    df['medium_term_signal'] = (
        df['price_trend_score'] * 0.4 +
        df['volume_price_alignment'] * 0.3 +
        df['volatility_stability'] * 0.3
    )
    
    # Long-term context (20+ days)
    # Structural levels
    df['ma_20'] = df['close'].rolling(20).mean()
    df['support_resistance_score'] = (
        (df['close'] > df['ma_20']).astype(int) * 2 - 1
    ) * (df['close'] / df['ma_20'] - 1)
    
    # Fundamental liquidity conditions
    df['amount_ma_20'] = df['amount'].rolling(20).mean()
    df['liquidity_trend'] = df['amount'] / df['amount_ma_20']
    
    # Market regime characteristics
    df['volatility_20d'] = df['returns_1d'].rolling(20).std()
    df['volatility_regime'] = df['volatility_20d'] / df['volatility_20d'].rolling(40).mean()
    
    # Long-term signal
    df['long_term_signal'] = (
        df['support_resistance_score'] * 0.5 +
        df['liquidity_trend'] * 0.3 +
        (1 / df['volatility_regime']) * 0.2
    )
    
    # Timeframe alignment score
    df['timeframe_alignment'] = (
        (np.sign(df['short_term_signal']) == np.sign(df['medium_term_signal'])).astype(int) +
        (np.sign(df['short_term_signal']) == np.sign(df['long_term_signal'])).astype(int) +
        (np.sign(df['medium_term_signal']) == np.sign(df['long_term_signal'])).astype(int)
    ) / 3.0
    
    # Signal strength measurement
    df['signal_strength'] = (
        np.abs(df['short_term_signal']) * 0.4 +
        np.abs(df['medium_term_signal']) * 0.35 +
        np.abs(df['long_term_signal']) * 0.25
    )
    
    # Volatility regime detection for dynamic weighting
    df['current_volatility'] = df['returns_1d'].rolling(5).std()
    df['volatility_ratio'] = df['current_volatility'] / df['current_volatility'].rolling(20).mean()
    
    # Dynamic weighting based on volatility regime
    def get_timeframe_weights(vol_ratio):
        if vol_ratio > 1.2:  # High volatility
            return [0.2, 0.4, 0.4]  # Emphasize medium and long-term
        elif vol_ratio < 0.8:  # Low volatility
            return [0.5, 0.3, 0.2]  # Emphasize short-term
        else:  # Normal volatility
            return [0.4, 0.35, 0.25]
    
    # Apply dynamic weighting
    final_factor = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if pd.notna(df['volatility_ratio'].iloc[i]):
            weights = get_timeframe_weights(df['volatility_ratio'].iloc[i])
            aligned_signal = (
                df['short_term_signal'].iloc[i] * weights[0] +
                df['medium_term_signal'].iloc[i] * weights[1] +
                df['long_term_signal'].iloc[i] * weights[2]
            )
            # Apply timeframe alignment multiplier
            final_factor.iloc[i] = aligned_signal * (1 + df['timeframe_alignment'].iloc[i])
        else:
            final_factor.iloc[i] = np.nan
    
    # Risk-adjusted output with volatility scaling
    volatility_scaling = 1 / (1 + df['current_volatility'])
    final_factor = final_factor * volatility_scaling
    
    return final_factor
