import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Fractal Momentum Efficiency Factor combining multi-scale momentum quality,
    volume-price fracture dynamics, volatility asymmetry, and microstructure efficiency
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # 1. Multi-Scale Momentum Quality
    # Momentum acceleration (3d vs 8d velocity change)
    mom_3d = df['close'].pct_change(3)
    mom_8d = df['close'].pct_change(8)
    momentum_acceleration = mom_3d - mom_8d
    
    # Momentum persistence (autocorrelation over 5 days)
    momentum_persistence = df['close'].pct_change().rolling(5).apply(
        lambda x: x.autocorr() if len(x) == 5 and not x.isna().any() else 0, raw=False
    )
    
    # 2. Volume-Price Fracture Dynamics
    # Gap fracture intensity
    gap_fracture = (df['high'] + df['low'] - 2 * df['close']) / (df['high'] - df['low'] + 1e-8)
    
    # Volume-weighted fracture across windows
    volume_weighted_fracture_3d = (gap_fracture * df['volume']).rolling(3).mean()
    volume_weighted_fracture_8d = (gap_fracture * df['volume']).rolling(8).mean()
    fracture_alignment = volume_weighted_fracture_3d - volume_weighted_fracture_8d
    
    # Intraday efficiency ratio
    intraday_efficiency = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    
    # 3. Volatility Asymmetry & Fracture Regimes
    # Directional volatility structure
    returns = df['close'].pct_change()
    upside_vol = returns[returns > 0].rolling(10, min_periods=5).std()
    downside_vol = (-returns[returns < 0]).rolling(10, min_periods=5).std()
    volatility_asymmetry = upside_vol / (downside_vol + 1e-8)
    volatility_asymmetry = volatility_asymmetry.reindex(df.index).fillna(1.0)
    
    # 4. Range Efficiency & Breakout Validation
    pre_close = df['close'].shift(1)
    gap_efficiency = (df['open'] - pre_close) / (df['high'] - df['low'] + 1e-8)
    gap_absorption = (df['high'] - df['low']) / (abs(df['open'] - pre_close) + 1e-8)
    
    # 5. Fractal Flow Accumulation Patterns
    directional_flow = (df['close'] - df['open']) * df['volume']
    flow_3d = directional_flow.rolling(3).sum()
    flow_6d = directional_flow.rolling(6).sum()
    cumulative_flow_ratio = flow_3d / (flow_6d + 1e-8)
    
    # 6. Asymmetric Microstructure Efficiency
    # Directional price efficiency
    up_days = df['close'] > df['open']
    down_days = df['close'] < df['open']
    
    up_day_efficiency = intraday_efficiency[up_days].rolling(5).mean()
    down_day_efficiency = intraday_efficiency[down_days].rolling(5).mean()
    efficiency_asymmetry = up_day_efficiency / (down_day_efficiency + 1e-8)
    efficiency_asymmetry = efficiency_asymmetry.reindex(df.index).fillna(1.0)
    
    # Volume-weighted fracture curvature
    volume_weighted_fracture = (gap_fracture * df['volume']).rolling(5).mean()
    fracture_curvature = volume_weighted_fracture.diff(2)
    
    # 7. Fractal Flow Imbalance Framework
    # Directional volume flow
    up_volume = df['volume'].where(up_days, 0)
    down_volume = df['volume'].where(down_days, 0)
    flow_imbalance = (up_volume.rolling(3).sum() - down_volume.rolling(3).sum()) / (df['volume'].rolling(3).sum() + 1e-8)
    
    # Flow momentum
    flow_momentum = flow_imbalance.diff(3)
    
    # 8. Cross-Feature Integration & Regime Adaptation
    # Volatility-weighted momentum quality
    volatility_weighted_momentum = momentum_acceleration / (df['close'].pct_change().rolling(10).std() + 1e-8)
    
    # Microstructure-flow-fracture convergence
    microstructure_convergence = (
        efficiency_asymmetry.rank(pct=True) + 
        flow_imbalance.rank(pct=True) + 
        gap_fracture.abs().rank(pct=True)
    ) / 3
    
    # Volatility-liquidity-fracture integration
    volatility_liquidity_integration = (
        volatility_asymmetry.rank(pct=True) + 
        (1 / (df['volume'].rolling(10).std() + 1e-8)).rank(pct=True) + 
        gap_fracture.abs().rank(pct=True)
    ) / 3
    
    # Final factor integration with regime adaptation
    # Trending regime components
    trending_components = (
        momentum_persistence.rank(pct=True) + 
        flow_imbalance.rank(pct=True) + 
        microstructure_convergence
    ) / 3
    
    # Mean-reverting regime components  
    mean_reverting_components = (
        gap_fracture.abs().rank(pct=True) + 
        (1 / gap_absorption).rank(pct=True) + 
        fracture_curvature.abs().rank(pct=True)
    ) / 3
    
    # Volatility regime components
    volatility_components = (
        volatility_asymmetry.rank(pct=True) + 
        volatility_liquidity_integration + 
        flow_momentum.abs().rank(pct=True)
    ) / 3
    
    # Dynamic regime weighting based on recent market conditions
    recent_volatility = df['close'].pct_change().rolling(5).std()
    recent_trend_strength = df['close'].pct_change(5).abs()
    
    # Regime classification weights
    trending_weight = (recent_trend_strength / (recent_volatility + 1e-8)).clip(0, 2)
    mean_reverting_weight = (1 / (gap_absorption + 1e-8)).clip(0, 2)
    volatility_weight = recent_volatility.rank(pct=True)
    
    # Normalize weights
    total_weight = trending_weight + mean_reverting_weight + volatility_weight + 1e-8
    trending_weight = trending_weight / total_weight
    mean_reverting_weight = mean_reverting_weight / total_weight
    volatility_weight = volatility_weight / total_weight
    
    # Final factor with regime-adaptive weighting
    fractal_momentum_efficiency = (
        trending_weight * trending_components +
        mean_reverting_weight * mean_reverting_components +
        volatility_weight * volatility_components
    )
    
    # Apply multi-timeframe consistency validation
    short_term = fractal_momentum_efficiency.rolling(3).mean()
    medium_term = fractal_momentum_efficiency.rolling(8).mean()
    consistency_score = 1 - abs(short_term - medium_term) / (abs(medium_term) + 1e-8)
    
    # Final factor with consistency adjustment
    final_factor = fractal_momentum_efficiency * consistency_score
    
    return final_factor
