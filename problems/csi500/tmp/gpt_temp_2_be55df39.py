import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Scale Price Momentum
    data['mom_3'] = data['close'] / data['close'].shift(3) - 1
    data['mom_8'] = data['close'] / data['close'].shift(8) - 1
    data['mom_21'] = data['close'] / data['close'].shift(21) - 1
    
    # Momentum Scaling Patterns
    data['short_med_ratio'] = np.where(data['mom_8'] != 0, data['mom_3'] / data['mom_8'], 0)
    data['med_long_ratio'] = np.where(data['mom_21'] != 0, data['mom_8'] / data['mom_21'], 0)
    
    # Fractal Consistency Score
    data['pos_momentum_count'] = ((data['mom_3'] > 0).astype(int) + 
                                 (data['mom_8'] > 0).astype(int) + 
                                 (data['mom_21'] > 0).astype(int))
    data['neg_momentum_count'] = ((data['mom_3'] < 0).astype(int) + 
                                 (data['mom_8'] < 0).astype(int) + 
                                 (data['mom_21'] < 0).astype(int))
    data['momentum_consistency'] = data['pos_momentum_count'] - data['neg_momentum_count']
    
    # Multi-Scale Volume Patterns
    data['vol_ratio_3'] = data['volume'] / data['volume'].shift(3)
    data['vol_ratio_8'] = data['volume'] / data['volume'].shift(8)
    data['vol_ratio_21'] = data['volume'] / data['volume'].shift(21)
    
    # Volume Momentum Alignment
    vol_momentum_alignment = []
    for i in range(len(data)):
        if i < 21:
            vol_momentum_alignment.append(0)
            continue
            
        vol_trend_3 = 1 if data['vol_ratio_3'].iloc[i] > 1 else -1 if data['vol_ratio_3'].iloc[i] < 1 else 0
        vol_trend_8 = 1 if data['vol_ratio_8'].iloc[i] > 1 else -1 if data['vol_ratio_8'].iloc[i] < 1 else 0
        vol_trend_21 = 1 if data['vol_ratio_21'].iloc[i] > 1 else -1 if data['vol_ratio_21'].iloc[i] < 1 else 0
        
        price_trend_3 = 1 if data['mom_3'].iloc[i] > 0 else -1 if data['mom_3'].iloc[i] < 0 else 0
        price_trend_8 = 1 if data['mom_8'].iloc[i] > 0 else -1 if data['mom_8'].iloc[i] < 0 else 0
        price_trend_21 = 1 if data['mom_21'].iloc[i] > 0 else -1 if data['mom_21'].iloc[i] < 0 else 0
        
        alignment_score = (vol_trend_3 * price_trend_3 + 
                         vol_trend_8 * price_trend_8 + 
                         vol_trend_21 * price_trend_21)
        vol_momentum_alignment.append(alignment_score)
    
    data['vol_momentum_alignment'] = vol_momentum_alignment
    
    # Cross-Scale Momentum Divergence
    data['short_med_divergence'] = data['mom_3'] - data['mom_8']
    data['med_long_divergence'] = data['mom_8'] - data['mom_21']
    
    # Fractal Divergence Score
    data['divergence_score'] = (2 * data['short_med_divergence'] + 
                              1.5 * data['med_long_divergence'])
    
    # Multi-Scale Range Analysis
    data['range_3'] = data['high'].rolling(window=3).max() - data['low'].rolling(window=3).min()
    data['range_8'] = data['high'].rolling(window=8).max() - data['low'].rolling(window=8).min()
    data['range_21'] = data['high'].rolling(window=21).max() - data['low'].rolling(window=21).min()
    
    # Efficiency Ratios
    data['net_move_3'] = (data['close'] - data['close'].shift(3)).abs()
    data['net_move_8'] = (data['close'] - data['close'].shift(8)).abs()
    data['net_move_21'] = (data['close'] - data['close'].shift(21)).abs()
    
    data['efficiency_3'] = np.where(data['range_3'] > 0, data['net_move_3'] / data['range_3'], 0)
    data['efficiency_8'] = np.where(data['range_8'] > 0, data['net_move_8'] / data['range_8'], 0)
    data['efficiency_21'] = np.where(data['range_21'] > 0, data['net_move_21'] / data['range_21'], 0)
    
    # Efficiency Scaling Relationships
    data['eff_short_med'] = np.where(data['efficiency_8'] != 0, data['efficiency_3'] / data['efficiency_8'], 0)
    data['eff_med_long'] = np.where(data['efficiency_21'] != 0, data['efficiency_8'] / data['efficiency_21'], 0)
    
    # Volatility Fractal Structure
    data['volatility_3'] = data['mom_3'].rolling(window=3).std()
    data['volatility_8'] = data['mom_8'].rolling(window=8).std()
    data['volatility_21'] = data['mom_21'].rolling(window=21).std()
    
    # Composite Fractal Alpha Calculation
    alpha_components = []
    
    for i in range(len(data)):
        if i < 21:
            alpha_components.append(0)
            continue
            
        # Core Fractal Momentum Component (40% weight)
        momentum_score = (0.4 * data['mom_3'].iloc[i] + 
                        0.35 * data['mom_8'].iloc[i] + 
                        0.25 * data['mom_21'].iloc[i])
        
        # Volume Fractal Confirmation (25% weight)
        volume_score = (0.1 * data['vol_momentum_alignment'].iloc[i] + 
                      0.08 * data['vol_ratio_3'].iloc[i] + 
                      0.07 * data['vol_ratio_8'].iloc[i])
        
        # Fractal Efficiency Enhancement (20% weight)
        efficiency_score = (0.08 * data['efficiency_3'].iloc[i] + 
                         0.07 * data['efficiency_8'].iloc[i] + 
                         0.05 * data['efficiency_21'].iloc[i])
        
        # Regime-Aware Adjustments (15% weight)
        volatility_regime = np.mean([data['volatility_3'].iloc[i], 
                                   data['volatility_8'].iloc[i], 
                                   data['volatility_21'].iloc[i]])
        
        trend_strength = np.mean([abs(data['mom_3'].iloc[i]), 
                                abs(data['mom_8'].iloc[i]), 
                                abs(data['mom_21'].iloc[i])])
        
        regime_score = 0.1 * (1 - volatility_regime) + 0.05 * trend_strength
        
        # Final Composite Alpha
        composite_alpha = (0.4 * momentum_score + 
                         0.25 * volume_score + 
                         0.2 * efficiency_score + 
                         0.15 * regime_score)
        
        alpha_components.append(composite_alpha)
    
    alpha_series = pd.Series(alpha_components, index=data.index, name='fractal_alpha')
    
    return alpha_series
