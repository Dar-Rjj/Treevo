import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate daily price efficiency
    data['efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # 1. Asymmetric Price Efficiency Momentum Calculation
    # Short-term efficiency momentum (3-day)
    data['efficiency_momentum_3d'] = data['efficiency'] / data['efficiency'].shift(3) - 1
    data['price_momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['short_term_efficiency'] = data['efficiency_momentum_3d'] * data['price_momentum_3d']
    
    # Medium-term efficiency trend (8-day)
    data['efficiency_8d_avg'] = data['efficiency'].rolling(window=8).mean()
    data['efficiency_20d_avg'] = data['efficiency'].rolling(window=20).mean()
    data['efficiency_deviation'] = data['efficiency_8d_avg'] - data['efficiency_20d_avg']
    
    # Efficiency persistence
    data['efficiency_persistence'] = data['efficiency'].rolling(window=5).apply(
        lambda x: np.sum(np.sign(x) == np.sign(x.iloc[-1])) / len(x) if len(x) == 5 else np.nan
    )
    
    # Ultra-short efficiency patterns
    data['morning_efficiency'] = (data['high'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['afternoon_efficiency'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    data['intraday_consistency'] = data['morning_efficiency'] * data['afternoon_efficiency']
    
    # Asymmetric Volatility Integration
    # Calculate efficiency volatility asymmetry
    def calc_efficiency_volatility_ratio(efficiency_series, window=21):
        positive_efficiency = efficiency_series.where(efficiency_series > 0)
        negative_efficiency = efficiency_series.where(efficiency_series < 0)
        
        positive_std = positive_efficiency.rolling(window=window).std()
        negative_std = negative_efficiency.rolling(window=window).std()
        
        return positive_std / negative_std.replace(0, np.nan)
    
    data['efficiency_vol_ratio'] = calc_efficiency_volatility_ratio(data['efficiency'])
    
    # Volatility momentum
    efficiency_vol = data['efficiency'].rolling(window=5).std()
    data['volatility_momentum'] = efficiency_vol / efficiency_vol.shift(10)
    
    # Combined Efficiency Momentum Factor
    data['efficiency_momentum'] = (
        data['short_term_efficiency'] * data['efficiency_vol_ratio'] * 
        data['efficiency_deviation'] * data['intraday_consistency']
    )
    
    # 2. Volume-Structure Convergence Assessment
    # Volume-weighted efficiency analysis
    data['fast_vw_efficiency'] = (
        (data['efficiency'] * data['volume']).rolling(window=8).sum() / 
        data['volume'].rolling(window=8).sum()
    )
    data['slow_vw_efficiency'] = (
        (data['efficiency'] * data['volume']).rolling(window=21).sum() / 
        data['volume'].rolling(window=21).sum()
    )
    data['vw_efficiency_divergence'] = data['fast_vw_efficiency'] - data['slow_vw_efficiency']
    
    # Micro-structure confirmation patterns
    data['trade_size'] = data['amount'] / data['volume'].replace(0, np.nan)
    
    # Morning efficiency strength persistence
    data['morning_persistence'] = data['morning_efficiency'].rolling(window=5).apply(
        lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) == 5 and not np.isnan(x).any() else np.nan
    )
    
    # Afternoon recovery consistency
    data['afternoon_consistency'] = data['afternoon_efficiency'].rolling(window=5).std()
    
    # Convergence strength measurement
    data['volume_concentration'] = (
        data['volume'] / data['volume'].rolling(window=21).mean() * 
        np.abs(data['efficiency'] / data['efficiency'].rolling(window=21).std())
    )
    
    # Volume-efficiency correlation
    data['volume_efficiency_corr'] = data['volume'].rolling(window=10).corr(data['efficiency'])
    
    # 3. Regime-Adaptive Signal Enhancement
    # Efficiency volatility regime
    efficiency_vol_10d = data['efficiency'].rolling(window=10).std()
    data['efficiency_vol_regime'] = pd.cut(
        efficiency_vol_10d, 
        bins=[0, efficiency_vol_10d.quantile(0.33), efficiency_vol_10d.quantile(0.67), float('inf')],
        labels=[0, 1, 2]
    ).astype(float)
    
    # Volume regime assessment
    volume_std_15d = data['volume'].rolling(window=15).std()
    data['volume_surge'] = data['volume'] / data['volume'].rolling(window=21).mean()
    data['volume_regime'] = pd.cut(
        data['volume_surge'],
        bins=[0, 0.8, 1.2, float('inf')],
        labels=[0, 1, 2]
    ).astype(float)
    
    # Combined regime analysis
    data['combined_regime'] = data['efficiency_vol_regime'] * data['volume_regime']
    
    # Regime-specific adjustments
    high_vol_regime_mask = data['efficiency_vol_regime'] == 2
    low_vol_regime_mask = data['efficiency_vol_regime'] == 0
    
    # Apply regime-specific weights
    regime_weight = np.where(high_vol_regime_mask, 1.5, 
                           np.where(low_vol_regime_mask, 0.8, 1.0))
    
    # 4. Composite Alpha Factor Generation
    # Core efficiency-volume convergence composite
    efficiency_volume_convergence = (
        data['efficiency_momentum'] * 
        data['vw_efficiency_divergence'] * 
        data['volume_efficiency_corr']
    )
    
    # Multi-dimensional regime adjustment
    regime_adjusted = efficiency_volume_convergence * regime_weight
    
    # Structure-enhanced timing refinement
    structure_refinement = (
        data['morning_persistence'] * 
        (1 / (data['afternoon_consistency'] + 0.1)) * 
        data['trade_size']
    )
    
    # Final adaptive integration
    alpha_factor = (
        regime_adjusted * 
        structure_refinement * 
        data['volume_concentration'] * 
        data['efficiency_persistence']
    )
    
    # Apply divergence penalty
    negative_corr_mask = data['volume_efficiency_corr'] < 0
    alpha_factor = np.where(negative_corr_mask, alpha_factor * 0.5, alpha_factor)
    
    # Multi-timeframe consistency score
    short_term_eff = data['efficiency'].rolling(window=3).mean()
    medium_term_eff = data['efficiency'].rolling(window=8).mean()
    consistency_score = np.where(
        np.sign(short_term_eff) == np.sign(medium_term_eff), 1.2, 0.8
    )
    
    # Final weighted factor
    final_factor = alpha_factor * consistency_score
    
    return final_factor
