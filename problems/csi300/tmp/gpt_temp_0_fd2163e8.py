import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Scale Momentum Dynamics
    # Momentum velocity (5-day return minus previous 5-day return)
    mom_5d = data['close'].pct_change(5)
    mom_velocity = mom_5d - mom_5d.shift(5)
    
    # Momentum persistence (consecutive days with same return direction)
    returns = data['close'].pct_change()
    mom_direction = np.sign(returns)
    mom_persistence = mom_direction.rolling(window=5, min_periods=1).apply(
        lambda x: len([i for i in range(1, len(x)) if x.iloc[i] == x.iloc[i-1] and x.iloc[i] != 0]), 
        raw=False
    )
    
    # Momentum curvature (10-day return minus average of 5-day and 20-day returns)
    mom_10d = data['close'].pct_change(10)
    mom_20d = data['close'].pct_change(20)
    mom_curvature = mom_10d - (mom_5d + mom_20d) / 2
    
    # Momentum regime (ratio of short-term to medium-term momentum)
    mom_regime = mom_5d / mom_20d.replace(0, np.nan)
    
    # Volume Flow Characterization
    # Volume intensity (current volume / 20-day median volume)
    vol_median_20d = data['volume'].rolling(window=20, min_periods=1).median()
    vol_intensity = data['volume'] / vol_median_20d
    
    # Volume directional persistence (consecutive days with volume above average)
    vol_avg_20d = data['volume'].rolling(window=20, min_periods=1).mean()
    vol_above_avg = (data['volume'] > vol_avg_20d).astype(int)
    vol_persistence = vol_above_avg.rolling(window=5, min_periods=1).apply(
        lambda x: len([i for i in range(1, len(x)) if x.iloc[i] == 1 and x.iloc[i] == x.iloc[i-1]]), 
        raw=False
    )
    
    # Volume-price efficiency (return per unit volume)
    vol_efficiency = returns / (data['volume'].replace(0, np.nan))
    
    # Volume clustering (ratio of high volume days to total days in lookback)
    high_vol_threshold = data['volume'].rolling(window=20, min_periods=1).quantile(0.7)
    high_vol_days = (data['volume'] > high_vol_threshold).rolling(window=10, min_periods=1).sum()
    vol_clustering = high_vol_days / 10
    
    # Price Structure Analysis
    # Relative position in daily range ((Close - Low) / (High - Low))
    daily_range = data['high'] - data['low']
    range_position = (data['close'] - data['low']) / daily_range.replace(0, np.nan)
    
    # Range expansion/contraction (current range vs 5-day average range)
    avg_range_5d = daily_range.rolling(window=5, min_periods=1).mean()
    range_expansion = daily_range / avg_range_5d
    
    # Gap dynamics (open vs previous close relationship)
    gap_dynamics = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Support/resistance proximity (distance from recent highs/lows)
    recent_high_20d = data['high'].rolling(window=20, min_periods=1).max()
    recent_low_20d = data['low'].rolling(window=20, min_periods=1).min()
    support_proximity = (data['close'] - recent_low_20d) / (recent_high_20d - recent_low_20d).replace(0, np.nan)
    
    # Adaptive Signal Convergence
    # Momentum-volume synchronization (momentum direction × volume intensity sign)
    mom_vol_sync = np.sign(mom_5d) * np.sign(vol_intensity - 1)
    
    # Structure-momentum alignment (range position × momentum velocity)
    structure_mom_align = range_position * mom_velocity
    
    # Volume-structure efficiency (volume intensity × range position)
    vol_structure_eff = vol_intensity * range_position
    
    # Multi-scale convergence (agreement across different momentum timeframes)
    mom_agreement = ((np.sign(mom_5d) == np.sign(mom_10d)) & 
                    (np.sign(mom_10d) == np.sign(mom_20d))).astype(int)
    
    # Regime-Adaptive Weighting
    # High momentum regime: emphasize velocity and persistence
    high_mom_regime = (mom_5d.abs() > mom_5d.rolling(window=20, min_periods=1).std()).astype(int)
    
    # Low momentum regime: focus on structure and volume efficiency
    low_mom_regime = (mom_5d.abs() < mom_5d.rolling(window=20, min_periods=1).std() * 0.5).astype(int)
    
    # High volume intensity: prioritize volume-driven signals
    high_vol_regime = (vol_intensity > 1.5).astype(int)
    
    # Range expansion: weight structure-based indicators higher
    range_exp_regime = (range_expansion > 1.2).astype(int)
    
    # Composite Factor Generation
    # Base components
    momentum_component = (mom_velocity * 0.3 + mom_persistence * 0.2 + 
                         mom_curvature * 0.25 + mom_regime * 0.25)
    
    volume_component = (vol_intensity * 0.3 + vol_persistence * 0.2 + 
                       vol_efficiency * 0.25 + vol_clustering * 0.25)
    
    structure_component = (range_position * 0.3 + range_expansion * 0.25 + 
                          gap_dynamics * 0.2 + support_proximity * 0.25)
    
    convergence_component = (mom_vol_sync * 0.3 + structure_mom_align * 0.25 + 
                            vol_structure_eff * 0.25 + mom_agreement * 0.2)
    
    # Regime-adaptive weighting
    regime_weights = pd.DataFrame({
        'high_mom': high_mom_regime,
        'low_mom': low_mom_regime,
        'high_vol': high_vol_regime,
        'range_exp': range_exp_regime
    })
    
    # Calculate regime-specific factors
    high_mom_factor = (momentum_component * 0.5 + convergence_component * 0.3 + 
                      volume_component * 0.1 + structure_component * 0.1)
    
    low_mom_factor = (structure_component * 0.4 + volume_component * 0.3 + 
                     convergence_component * 0.2 + momentum_component * 0.1)
    
    high_vol_factor = (volume_component * 0.5 + convergence_component * 0.25 + 
                      momentum_component * 0.15 + structure_component * 0.1)
    
    range_exp_factor = (structure_component * 0.45 + convergence_component * 0.25 + 
                       momentum_component * 0.15 + volume_component * 0.15)
    
    # Default factor (balanced)
    default_factor = (momentum_component * 0.25 + volume_component * 0.25 + 
                     structure_component * 0.25 + convergence_component * 0.25)
    
    # Combine regime-specific factors
    composite_factor = (
        regime_weights['high_mom'] * high_mom_factor +
        regime_weights['low_mom'] * low_mom_factor +
        regime_weights['high_vol'] * high_vol_factor +
        regime_weights['range_exp'] * range_exp_factor +
        ((1 - regime_weights.max(axis=1)) * default_factor)
    )
    
    # Normalize and handle NaN values
    composite_factor = (composite_factor - composite_factor.rolling(window=20, min_periods=1).mean()) / composite_factor.rolling(window=20, min_periods=1).std()
    composite_factor = composite_factor.fillna(0)
    
    return composite_factor
