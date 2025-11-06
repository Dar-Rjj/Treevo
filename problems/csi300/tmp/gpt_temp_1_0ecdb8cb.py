import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Asset Momentum Diffusion with Liquidity Barrier Dynamics
    Combines momentum transmission patterns with liquidity barrier strength
    to predict future stock returns.
    """
    # Initialize output series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # Calculate daily returns and price-based features
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['prev_close'] = df['close'].shift(1)
    df['overnight_gap'] = (df['open'] - df['prev_close']).abs() / df['prev_close']
    df['intraday_range'] = (df['high'] - df['low']) / df['open']
    df['close_to_mid'] = (df['close'] - (df['high'] + df['low']) / 2).abs() / (df['high'] - df['low'])
    
    # Momentum structure components
    df['momentum_5d'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_10d'] = df['close'] / df['close'].shift(10) - 1
    df['momentum_ratio'] = df['momentum_5d'] / (df['momentum_10d'] + 1e-8)
    
    # Liquidity barrier components
    df['avg_volume_5d'] = df['volume'].rolling(window=5).mean()
    df['volume_ratio'] = df['volume'] / (df['avg_volume_5d'] + 1e-8)
    
    # Directional liquidity strength
    df['up_day'] = (df['close'] > df['open']).astype(int)
    df['down_day'] = (df['close'] < df['open']).astype(int)
    
    # Calculate rolling directional volume ratios
    up_volume_5d = (df['volume'] * df['up_day']).rolling(window=5).sum()
    down_volume_5d = (df['volume'] * df['down_day']).rolling(window=5).sum()
    df['up_liquidity_ratio'] = up_volume_5d / (up_volume_5d + down_volume_5d + 1e-8)
    df['down_liquidity_ratio'] = down_volume_5d / (up_volume_5d + down_volume_5d + 1e-8)
    df['liquidity_asymmetry'] = df['up_liquidity_ratio'] / (df['down_liquidity_ratio'] + 1e-8)
    
    # Price impact efficiency (proxy for bid-ask spread)
    df['price_impact'] = (df['high'] - df['low']).abs() / (df['volume'] + 1e-8)
    df['price_impact_ma'] = df['price_impact'].rolling(window=5).mean()
    df['price_impact_ratio'] = df['price_impact'] / (df['price_impact_ma'] + 1e-8)
    
    # Multi-scale momentum efficiency with liquidity completion
    df['opening_efficiency'] = df['overnight_gap'] / (df['volume_ratio'] + 1e-8)
    df['midday_efficiency'] = df['intraday_range'] / (df['close_to_mid'] + 1e-8)
    df['closing_efficiency'] = df['close_to_mid'] / (df['volume_ratio'] + 1e-8)
    
    # Momentum-liquidity interaction components
    df['momentum_liquidity_alignment'] = df['momentum_5d'] * df['liquidity_asymmetry']
    df['volume_momentum_efficiency'] = df['momentum_5d'] / (df['price_impact_ratio'] + 1e-8)
    
    # Regime detection signals
    df['momentum_persistence'] = (df['momentum_5d'] > 0) & (df['momentum_10d'] > 0)
    df['liquidity_concentration'] = df['volume_ratio'] > 1.2
    df['high_momentum_regime'] = (df['momentum_5d'] > df['momentum_5d'].rolling(window=20).quantile(0.7))
    df['low_momentum_regime'] = (df['momentum_5d'] < df['momentum_5d'].rolling(window=20).quantile(0.3))
    
    # Cross-asset momentum transmission timing (using sector-like momentum)
    sector_momentum = df['close'].pct_change().rolling(window=5).mean()
    df['sector_momentum_diffusion'] = df['momentum_5d'] - sector_momentum
    df['momentum_transmission_timing'] = df['sector_momentum_diffusion'] * df['liquidity_asymmetry']
    
    # Liquidity barrier strength scoring
    barrier_strength = (
        df['up_liquidity_ratio'].rolling(window=10).std() + 
        df['price_impact_ratio'].rolling(window=10).std() +
        df['volume_ratio'].rolling(window=10).std()
    )
    df['liquidity_barrier_score'] = barrier_strength / barrier_strength.rolling(window=20).mean()
    
    # Composite alpha generation
    for i in range(len(df)):
        if i < 20:  # Ensure sufficient history
            alpha.iloc[i] = 0
            continue
            
        row = df.iloc[i]
        
        # Core alpha components
        momentum_timing = row['momentum_transmission_timing']
        liquidity_barrier = row['liquidity_barrier_score']
        interaction_efficiency = row['momentum_liquidity_alignment']
        
        # Regime-based weighting
        if row['high_momentum_regime']:
            regime_weight = 1.2
        elif row['low_momentum_regime']:
            regime_weight = 0.8
        else:
            regime_weight = 1.0
            
        # Calculate final alpha value
        alpha_value = (
            momentum_timing * 0.4 +
            liquidity_barrier * 0.3 +
            interaction_efficiency * 0.3
        ) * regime_weight
        
        alpha.iloc[i] = alpha_value
    
    # Normalize the alpha series
    alpha = (alpha - alpha.rolling(window=20).mean()) / (alpha.rolling(window=20).std() + 1e-8)
    
    return alpha
