import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate basic price-based features
    df = df.copy()
    
    # Daily Range Efficiency
    df['range_efficiency'] = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    
    # Multi-timeframe Range Efficiency Momentum
    df['re_momentum_5d'] = df['range_efficiency'] - df['range_efficiency'].shift(5)
    df['re_momentum_10d'] = df['range_efficiency'] - df['range_efficiency'].shift(10)
    df['re_momentum_20d'] = df['range_efficiency'] - df['range_efficiency'].shift(20)
    
    # Calculate returns for volatility
    df['returns'] = df['close'].pct_change()
    
    # Asymmetric Volatility
    returns_10d = df['returns'].rolling(window=10)
    df['upside_vol'] = returns_10d.apply(lambda x: np.std(x[x > 0]) if len(x[x > 0]) > 1 else np.nan)
    df['downside_vol'] = returns_10d.apply(lambda x: np.std(x[x < 0]) if len(x[x < 0]) > 1 else np.nan)
    df['vol_asymmetry'] = df['upside_vol'] / df['downside_vol'].replace(0, np.nan)
    
    # Scale range efficiency momentum by volatility asymmetry
    df['vol_adjusted_re_momentum'] = df['re_momentum_10d'] * df['vol_asymmetry']
    
    # Range Dynamics and Volatility Regime Detection
    df['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    
    # Rolling volatility from range dynamics
    df['range_vol_5d'] = df['true_range'].rolling(window=5).std()
    df['range_vol_10d'] = df['true_range'].rolling(window=10).std()
    df['range_vol_20d'] = df['true_range'].rolling(window=20).std()
    
    # Volatility regime shifts
    df['vol_compression'] = (df['range_vol_5d'] < df['range_vol_20d'] * 0.7).astype(int)
    df['vol_expansion'] = (df['range_vol_5d'] > df['range_vol_20d'] * 1.3).astype(int)
    
    # Range Expansion Momentum
    df['range_expansion_momentum'] = (df['true_range'] - df['true_range'].rolling(window=20).mean()) / df['true_range'].rolling(window=20).std()
    
    # Volume-Weighted Efficiency
    df['volume_efficiency'] = df['volume'] / (df['high'] - df['low']).replace(0, np.nan)
    
    # Rolling correlation between volume efficiency and price changes
    df['vol_eff_price_corr'] = df['volume_efficiency'].rolling(window=10).corr(df['returns'])
    
    # Directional Order Flow
    df['buying_day'] = (df['close'] > df['open']).astype(int)
    df['signed_volume'] = df['volume'] * np.where(df['buying_day'] == 1, 1, -1)
    df['order_flow_10d'] = df['signed_volume'].rolling(window=10).sum()
    
    # Volume momentum persistence
    df['volume_momentum'] = df['volume'] / df['volume'].rolling(window=10).mean()
    
    # Volume-weighted signal
    df['volume_weighted_signal'] = df['vol_adjusted_re_momentum'] * df['volume_efficiency'] * np.sign(df['order_flow_10d'])
    
    # Intraday Efficiency Patterns
    df['morning_strength'] = (df['high'] - df['open']) / df['open']
    df['afternoon_efficiency'] = (df['close'] - df['high']) / df['high']
    
    # Multi-timeframe intraday consistency
    df['intraday_consistency_5d'] = df['morning_strength'].rolling(window=5).corr(df['afternoon_efficiency'])
    df['intraday_stability_20d'] = df['range_efficiency'].rolling(window=20).std()
    
    # Efficiency Divergence
    df['short_medium_divergence'] = df['re_momentum_5d'] - df['re_momentum_10d']
    df['medium_long_divergence'] = df['re_momentum_10d'] - df['re_momentum_20d']
    df['divergence_momentum'] = (df['short_medium_divergence'] + df['medium_long_divergence']) / 2
    
    # Regime-Adaptive Integrated Signal Generation
    # Base signal from volatility-adjusted range efficiency
    base_signal = df['vol_adjusted_re_momentum']
    
    # Apply volume confirmation
    volume_confirmed_signal = base_signal * df['volume_weighted_signal']
    
    # Regime-specific adjustments
    # High volatility regime: emphasize range efficiency persistence
    high_vol_signal = np.where(
        df['vol_expansion'] == 1,
        volume_confirmed_signal * (1 + df['range_expansion_momentum']),
        volume_confirmed_signal
    )
    
    # Low volatility regime: focus on breakout potential
    low_vol_signal = np.where(
        df['vol_compression'] == 1,
        volume_confirmed_signal * (1 + abs(df['order_flow_10d'] / df['order_flow_10d'].rolling(window=20).std())),
        high_vol_signal
    )
    
    # Transition phases: incorporate divergence
    transition_mask = (df['vol_compression'].shift(1) != df['vol_compression']) | (df['vol_expansion'].shift(1) != df['vol_expansion'])
    final_signal = np.where(
        transition_mask,
        low_vol_signal + df['divergence_momentum'] * df['intraday_consistency_5d'],
        low_vol_signal
    )
    
    # Cross-timeframe consistency weighting
    timeframe_alignment = (
        np.sign(df['re_momentum_5d']) == np.sign(df['re_momentum_10d'])
    ) & (
        np.sign(df['re_momentum_10d']) == np.sign(df['re_momentum_20d'])
    )
    
    final_signal = final_signal * (1 + 0.2 * timeframe_alignment.astype(float))
    
    # Scale by regime change and range dynamics interaction
    regime_change_strength = abs(df['range_vol_5d'] - df['range_vol_20d']) / df['range_vol_20d']
    final_signal = final_signal * (1 + 0.1 * regime_change_strength)
    
    return pd.Series(final_signal, index=df.index)
