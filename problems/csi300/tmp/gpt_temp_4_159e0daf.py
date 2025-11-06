import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Novel alpha factor combining curvature-efficiency momentum, microstructure range breakout, 
    and liquidity-volatility convergence with regime-adaptive weighting.
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # 1. Curvature-Efficiency Momentum Component
    # Intraday curvature: deviation of mid-price from close
    df['mid_price'] = (df['high'] + df['low']) / 2
    df['curvature'] = (df['mid_price'] - df['close']) / (df['high'] - df['low']).replace(0, np.nan)
    
    # Volume-weighted curvature with clustering
    df['volume_cluster'] = df['volume'].rolling(window=5, min_periods=3).std() / df['volume'].rolling(window=5, min_periods=3).mean()
    df['curvature_weighted'] = df['curvature'] * df['volume_cluster']
    
    # Curvature persistence across timeframes
    df['curvature_3d_persistence'] = df['curvature_weighted'].rolling(window=3, min_periods=2).apply(
        lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 and not np.isnan(x).any() else 0
    )
    df['curvature_8d_persistence'] = df['curvature_weighted'].rolling(window=8, min_periods=5).apply(
        lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 and not np.isnan(x).any() else 0
    )
    
    # Directional efficiency
    df['directional_efficiency'] = ((df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)) * np.sign(df['close'] - df['open'])
    
    # Efficiency asymmetry (up vs down days)
    up_days = df['close'] > df['open']
    down_days = df['close'] < df['open']
    df['efficiency_up'] = df['directional_efficiency'].where(up_days, 0)
    df['efficiency_down'] = df['directional_efficiency'].where(down_days, 0)
    df['efficiency_asymmetry'] = (df['efficiency_up'].rolling(window=5, min_periods=3).mean() - 
                                 df['efficiency_down'].rolling(window=5, min_periods=3).mean())
    
    # Curvature-efficiency convergence
    df['curvature_efficiency_alignment'] = df['curvature_weighted'] * df['directional_efficiency']
    df['ce_convergence'] = df['curvature_efficiency_alignment'].rolling(window=5, min_periods=3).mean()
    
    # 2. Microstructure Range Breakout Component
    # Trade flow imbalance
    df['price_change'] = df['close'].pct_change()
    df['up_volume'] = df['volume'].where(df['price_change'] > 0, 0)
    df['down_volume'] = df['volume'].where(df['price_change'] < 0, 0)
    df['trade_flow'] = (df['up_volume'] - df['down_volume']) * df['price_change']
    
    # Trade flow momentum and acceleration
    df['flow_momentum_2d'] = df['trade_flow'].rolling(window=2, min_periods=2).mean()
    df['flow_momentum_5d'] = df['trade_flow'].rolling(window=5, min_periods=3).mean()
    df['flow_acceleration'] = df['flow_momentum_2d'] - df['flow_momentum_5d']
    
    # Range compression dynamics
    df['range_contraction'] = (df['high'] - df['low']) / (abs(df['close'] - df['open']).replace(0, np.nan))
    df['range_compression_3d'] = df['range_contraction'].rolling(window=3, min_periods=2).mean()
    
    # Volume acceleration for range expansion signals
    df['volume_acceleration'] = df['volume'].pct_change().rolling(window=3, min_periods=2).mean()
    
    # Microstructure range breakout signal
    df['micro_breakout'] = (df['range_compression_3d'].shift(1) * 
                           df['flow_acceleration'] * 
                           df['volume_acceleration'])
    
    # 3. Liquidity-Volatility Convergence Component
    # Volatility asymmetry
    df['upside_vol'] = (df['high'] - df['close']).rolling(window=5, min_periods=3).std()
    df['downside_vol'] = (df['close'] - df['low']).rolling(window=5, min_periods=3).std()
    df['vol_asymmetry'] = (df['upside_vol'] - df['downside_vol']) / (df['upside_vol'] + df['downside_vol']).replace(0, np.nan)
    
    # Liquidity provision asymmetry
    df['upper_range_volume'] = df['volume'].where((df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan) > 0.5, 0)
    df['lower_range_volume'] = df['volume'].where((df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan) <= 0.5, 0)
    df['liquidity_asymmetry'] = (df['upper_range_volume'].rolling(window=5, min_periods=3).sum() - 
                                df['lower_range_volume'].rolling(window=5, min_periods=3).sum()) / df['volume'].rolling(window=5, min_periods=3).sum().replace(0, np.nan)
    
    # Liquidity-volatility convergence
    df['liq_vol_convergence'] = df['liquidity_asymmetry'] * df['vol_asymmetry']
    
    # 4. Cross-Dimensional Regime Integration
    # Multi-feature convergence scoring
    df['convergence_score'] = (
        df['ce_convergence'].fillna(0) + 
        df['micro_breakout'].fillna(0) + 
        df['liq_vol_convergence'].fillna(0)
    )
    
    # Regime classification based on convergence strength
    df['convergence_std'] = df['convergence_score'].rolling(window=20, min_periods=10).std()
    df['convergence_mean'] = df['convergence_score'].rolling(window=20, min_periods=10).mean()
    df['regime_strength'] = abs(df['convergence_score'] - df['convergence_mean']) / df['convergence_std'].replace(0, np.nan)
    
    # Regime-adaptive weighting
    high_regime = df['regime_strength'] > 1.0
    moderate_regime = (df['regime_strength'] >= 0.5) & (df['regime_strength'] <= 1.0)
    low_regime = df['regime_strength'] < 0.5
    
    # Final alpha synthesis with regime-adaptive weights
    curvature_weight = pd.Series(np.where(high_regime, 0.4, np.where(moderate_regime, 0.3, 0.2)), index=df.index)
    micro_weight = pd.Series(np.where(high_regime, 0.35, np.where(moderate_regime, 0.4, 0.3)), index=df.index)
    liq_vol_weight = pd.Series(np.where(high_regime, 0.25, np.where(moderate_regime, 0.3, 0.5)), index=df.index)
    
    # Final alpha factor
    alpha = (
        curvature_weight * df['ce_convergence'].fillna(0) +
        micro_weight * df['micro_breakout'].fillna(0) +
        liq_vol_weight * df['liq_vol_convergence'].fillna(0)
    )
    
    # Normalize by recent volatility
    alpha_vol = alpha.rolling(window=20, min_periods=10).std().replace(0, np.nan)
    alpha = alpha / alpha_vol
    
    return alpha.fillna(0)
