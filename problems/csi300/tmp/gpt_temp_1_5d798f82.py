import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Association Momentum Divergence factor combining multi-asset correlation,
    volume-volatility dislocation, and temporal pattern asymmetry signals.
    """
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Required columns
    cols_needed = ['open', 'high', 'low', 'close', 'volume', 'amount']
    for col in cols_needed:
        if col not in df.columns:
            df[col] = 0.0
    
    # Calculate returns
    df['ret_1d'] = df['close'].pct_change()
    df['ret_5d'] = df['close'].pct_change(5)
    
    # Multi-Asset Correlation Structure
    # Sector-Relative Momentum (using market as proxy for sector)
    df['market_ret_5d'] = df['close'].pct_change(5).rolling(50, min_periods=20).mean()  # Market proxy
    df['sector_momentum_diff'] = df['ret_5d'] - df['market_ret_5d']
    
    # Market-Regime Correlation
    df['market_corr_20d'] = df['close'].pct_change().rolling(window=20).corr(df['close'].pct_change().rolling(window=5).mean())
    df['corr_regime'] = (df['market_corr_20d'] > df['market_corr_20d'].rolling(60).mean()).astype(int)
    
    # Cross-Instrument Price Alignment (using market as competitor proxy)
    df['market_strength'] = df['close'].pct_change(3) - df['close'].pct_change(3).rolling(20).mean()
    df['relative_strength_div'] = df['ret_5d'] - df['market_strength']
    
    # Volume-Volatility Dislocation
    # Abnormal Volume Patterns
    df['volume_percentile'] = df['volume'].rolling(60).apply(lambda x: (x.iloc[-1] > np.percentile(x[:-1], 80)) if len(x) > 1 else 0, raw=False)
    df['volume_spike_cluster'] = (df['volume'] > df['volume'].rolling(20).mean() * 1.5).rolling(3).sum()
    
    # Volatility Regime Mismatch
    df['stock_vol_10d'] = df['ret_1d'].rolling(10).std()
    df['market_vol_10d'] = df['ret_1d'].rolling(10).std().rolling(50).mean()  # Market vol proxy
    df['vol_regime_mismatch'] = (df['stock_vol_10d'] / df['market_vol_10d']) - 1
    
    # Price-Volume Efficiency Gap
    df['efficiency_daily'] = df['ret_1d'].abs() / (df['volume'] + 1e-8)
    df['efficiency_gap'] = df['efficiency_daily'] - df['efficiency_daily'].rolling(10).mean()
    
    # Temporal Pattern Asymmetry
    # Intraday Return Distribution (using open-to-close vs previous close-to-open)
    df['intraday_ret'] = (df['close'] - df['open']) / df['open']
    df['overnight_ret'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['morning_afternoon_asym'] = df['intraday_ret'] - df['overnight_ret'].abs()
    
    # Last Hour Momentum (using last 25% of trading range as proxy)
    df['range_high_low'] = (df['high'] - df['low']) / df['close']
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    df['last_hour_momentum'] = df['close_position'].diff(3)
    
    # Overnight-Intraday Return Disconnect
    df['overnight_intraday_disconnect'] = np.sign(df['overnight_ret']) != np.sign(df['intraday_ret'])
    df['disconnect_magnitude'] = df['overnight_ret'].abs() * df['overnight_intraday_disconnect']
    
    # Multi-Day Pattern Recognition
    df['ret_3d_seq'] = df['ret_1d'].rolling(3).apply(lambda x: 1 if (x > 0).all() else (-1 if (x < 0).all() else 0), raw=False)
    df['pattern_break'] = (df['ret_3d_seq'] != df['ret_3d_seq'].shift(1)).astype(int)
    
    # Signal Generation Framework
    # Cross-Asset Momentum Divergence
    df['cross_asset_signal'] = df['sector_momentum_diff'] * (1 + df['corr_regime'] * 0.5)
    
    # Volume-Volatility Signal Integration
    df['volume_vol_signal'] = (df['volume_percentile'] * df['vol_regime_mismatch'] * 
                              np.sign(df['efficiency_gap']))
    
    # Temporal Pattern Signals
    df['temporal_signal'] = (df['morning_afternoon_asym'] + 
                           df['last_hour_momentum'] - 
                           df['disconnect_magnitude'] * 2)
    
    # Composite Factor Construction
    # Regime-Adaptive Weighting
    high_corr_weight = df['corr_regime'] * 0.6 + 0.4
    low_corr_weight = (1 - df['corr_regime']) * 0.7 + 0.3
    vol_expansion_weight = (df['vol_regime_mismatch'] > 0).astype(float) * 0.5 + 0.5
    
    # Combine signals with regime weighting
    cross_asset_component = df['cross_asset_signal'] * high_corr_weight
    internal_pattern_component = df['temporal_signal'] * low_corr_weight
    volume_component = df['volume_vol_signal'] * vol_expansion_weight
    
    # Signal Conflict Resolution with majority voting
    signals = pd.DataFrame({
        'cross_asset': np.sign(cross_asset_component),
        'internal': np.sign(internal_pattern_component),
        'volume': np.sign(volume_component)
    })
    
    df['signal_agreement'] = signals.sum(axis=1)
    df['majority_signal'] = np.sign(df['signal_agreement'])
    
    # Dynamic Signal Persistence
    df['signal_strength'] = (abs(cross_asset_component) + 
                           abs(internal_pattern_component) + 
                           abs(volume_component)) / 3
    
    # Volume confirmation for signal extension
    df['volume_confirmation'] = (df['volume'] > df['volume'].rolling(10).mean()).astype(int)
    
    # Final composite factor
    df['composite_factor'] = (df['majority_signal'] * df['signal_strength'] * 
                            (1 + df['volume_confirmation'] * 0.3))
    
    # Apply pattern break detection for early exit
    df['final_factor'] = df['composite_factor'] * (1 - df['pattern_break'] * 0.5)
    
    result = df['final_factor']
    
    # Clean infinite values and handle NaNs
    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.fillna(method='ffill').fillna(0)
    
    return result
