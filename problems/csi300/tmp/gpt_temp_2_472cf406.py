import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Multi-Timeframe Pressure Accumulation
    # Calculate directional pressure
    df['pressure'] = (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low'])
    df['pressure'] = df['pressure'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Volume-weighted pressure accumulation
    df['vw_pressure_3d'] = (df['pressure'] * df['volume']).rolling(window=3, min_periods=1).sum() / df['volume'].rolling(window=3, min_periods=1).sum()
    df['vw_pressure_10d'] = (df['pressure'] * df['volume']).rolling(window=10, min_periods=1).sum() / df['volume'].rolling(window=10, min_periods=1).sum()
    
    # Pressure compression/expansion states
    df['pressure_std_5d'] = df['pressure'].rolling(window=5, min_periods=1).std()
    df['pressure_state'] = np.where(df['pressure_std_5d'] > df['pressure_std_5d'].rolling(window=20, min_periods=1).mean(), 'expansion', 'compression')
    
    # Volume-Validated Range Efficiency
    # Calculate intraday efficiency
    df['efficiency'] = (df['close'] - df['open']) / (df['high'] - df['low'])
    df['efficiency'] = df['efficiency'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Efficiency persistence
    df['efficiency_persistence'] = df['efficiency'].rolling(window=5, min_periods=1).apply(lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 and np.std(x) > 0 else 0)
    
    # Efficiency-momentum divergence
    df['efficiency_ma_5d'] = df['efficiency'].rolling(window=5, min_periods=1).mean()
    df['close_ma_5d'] = df['close'].rolling(window=5, min_periods=1).mean()
    df['efficiency_momentum_div'] = df['efficiency_ma_5d'] - df['efficiency_ma_5d'].shift(1) - (df['close_ma_5d'] - df['close_ma_5d'].shift(1))
    
    # Liquidity-Regime Classification
    # Calculate liquidity absorption
    df['liquidity_absorption'] = (df['high'] - df['low']) / df['volume']
    df['liquidity_absorption'] = df['liquidity_absorption'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Classify market liquidity states
    liquidity_ma = df['liquidity_absorption'].rolling(window=20, min_periods=1).mean()
    liquidity_std = df['liquidity_absorption'].rolling(window=20, min_periods=1).std()
    
    df['liquidity_regime'] = np.where(
        df['liquidity_absorption'] > liquidity_ma + liquidity_std, 'low',
        np.where(df['liquidity_absorption'] < liquidity_ma - liquidity_std, 'high', 'transition')
    )
    
    # Momentum-Quality Assessment
    # Calculate volume-scaled momentum
    df['momentum_5d_vol_scaled'] = (df['close'] - df['close'].shift(5)) * df['volume']
    df['momentum_10d_vol_scaled'] = (df['close'] - df['close'].shift(10)) * df['volume']
    
    # Pressure-momentum alignment strength
    df['pressure_momentum_corr_10d'] = df['pressure'].rolling(window=10, min_periods=1).corr(df['close'].pct_change().rolling(window=10, min_periods=1).mean())
    
    # Momentum exhaustion patterns
    df['momentum_exhaustion'] = (df['momentum_5d_vol_scaled'] - df['momentum_5d_vol_scaled'].shift(1)) / (df['momentum_5d_vol_scaled'].abs().rolling(window=5, min_periods=1).mean())
    
    # Adaptive Convergence Scoring
    # Combine pressure accumulation with momentum quality
    df['pressure_momentum_convergence'] = df['vw_pressure_10d'] * df['pressure_momentum_corr_10d'] * df['momentum_5d_vol_scaled']
    
    # Apply liquidity regime adjustments
    regime_multipliers = {
        'high': 1.2,    # Amplify signals in high liquidity
        'transition': 1.0,
        'low': 0.8      # Dampen signals in low liquidity
    }
    
    df['regime_adjusted_convergence'] = df['pressure_momentum_convergence'] * df['liquidity_regime'].map(regime_multipliers)
    
    # Incorporate efficiency persistence
    df['final_convergence_factor'] = (
        df['regime_adjusted_convergence'] * 
        (1 + df['efficiency_persistence']) * 
        (1 - np.abs(df['efficiency_momentum_div']))
    )
    
    # Normalize the final factor
    factor = df['final_convergence_factor'] / df['final_convergence_factor'].abs().rolling(window=20, min_periods=1).mean()
    
    return factor
