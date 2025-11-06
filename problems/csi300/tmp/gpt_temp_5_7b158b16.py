import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Microstructure Pressure-Flow Confluence factor
    """
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic components
    df['prev_close'] = df['close'].shift(1)
    df['prev_high'] = df['high'].shift(1)
    df['prev_low'] = df['low'].shift(1)
    df['prev_range'] = df['prev_high'] - df['prev_low']
    
    # True Range calculation
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['prev_close'])
    df['tr3'] = abs(df['low'] - df['prev_close'])
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Volatility regime detection (5-day rolling)
    df['volatility_regime'] = df['true_range'].rolling(window=5).mean() / df['true_range'].rolling(window=20).mean()
    
    # Pressure dynamics
    df['pressure'] = (df['close'] - df['open']) / np.where(df['high'] - df['low'] > 0, df['high'] - df['low'], 1)
    df['pressure_regime'] = df['pressure'].rolling(window=5).mean()
    
    # Range compression analysis
    df['range_ratio'] = (df['high'] - df['low']) / np.where(df['prev_range'] > 0, df['prev_range'], 1)
    df['compression_regime'] = df['range_ratio'].rolling(window=5).mean()
    
    # Volume analysis
    df['volume_ma5'] = df['volume'].rolling(window=5).mean()
    df['volume_ma20'] = df['volume'].rolling(window=20).mean()
    df['volume_acceleration'] = df['volume_ma5'] / np.where(df['volume_ma20'] > 0, df['volume_ma20'], 1)
    
    # Gap analysis
    df['gap_up'] = (df['open'] > df['prev_high']).astype(int)
    df['gap_down'] = (df['open'] < df['prev_low']).astype(int)
    df['gap_size'] = np.where(df['gap_up'] == 1, df['open'] - df['prev_high'],
                            np.where(df['gap_down'] == 1, df['prev_low'] - df['open'], 0))
    df['gap_fill_efficiency'] = np.where(df['gap_size'] > 0, 
                                       (df['close'] - df['open']) / df['gap_size'], 0)
    
    # Multi-dimensional regime classification
    df['vol_regime_class'] = np.where(df['volatility_regime'] < 0.8, 'low_vol',
                                    np.where(df['volatility_regime'] > 1.2, 'high_vol', 'normal_vol'))
    
    df['pressure_regime_class'] = np.where(df['pressure_regime'] > 0.3, 'high_pressure',
                                         np.where(df['pressure_regime'] < -0.3, 'low_pressure', 'neutral_pressure'))
    
    df['compression_class'] = np.where(df['compression_regime'] < 0.7, 'compressed',
                                     np.where(df['compression_regime'] > 1.3, 'expanded', 'normal_range'))
    
    # Pressure-Volume Elasticity
    df['pressure_volume_elasticity'] = (df['pressure'].rolling(window=5).std() + 1e-6) / \
                                     (df['volume_acceleration'].rolling(window=5).std() + 1e-6)
    
    # Flow synchronization across timeframes
    df['short_term_flow'] = df['close'].rolling(window=3).mean() - df['close'].rolling(window=8).mean()
    df['medium_term_flow'] = df['close'].rolling(window=8).mean() - df['close'].rolling(window=21).mean()
    df['flow_synchronization'] = np.sign(df['short_term_flow']) * np.sign(df['medium_term_flow'])
    
    # Regime-adaptive confluence scoring
    confluence_scores = []
    
    for i in range(len(df)):
        if i < 20:  # Ensure enough data for calculations
            confluence_scores.append(0)
            continue
            
        current = df.iloc[i]
        
        # Regime-based weighting
        vol_weight = 1.0
        if current['vol_regime_class'] == 'low_vol':
            vol_weight = 1.2
        elif current['vol_regime_class'] == 'high_vol':
            vol_weight = 0.8
            
        pressure_weight = 1.0
        if current['pressure_regime_class'] == 'high_pressure':
            pressure_weight = 1.3
        elif current['pressure_regime_class'] == 'low_pressure':
            pressure_weight = 0.7
            
        # Confluence components
        pressure_strength = current['pressure'] * pressure_weight
        
        # Volume confluence
        volume_confluence = current['volume_acceleration'] * np.sign(current['pressure'])
        
        # Flow synchronization score
        flow_score = current['flow_synchronization'] * abs(current['short_term_flow'])
        
        # Gap efficiency with regime context
        gap_score = 0
        if abs(current['gap_size']) > 0:
            gap_efficiency = current['gap_fill_efficiency']
            if current['vol_regime_class'] == 'low_vol' and current['compression_class'] == 'compressed':
                gap_score = gap_efficiency * 1.5
            else:
                gap_score = gap_efficiency
        
        # Elasticity adjustment
        elasticity_adjustment = np.clip(current['pressure_volume_elasticity'], 0.5, 2.0)
        
        # Final confluence calculation
        confluence = (pressure_strength * 0.4 + 
                     volume_confluence * 0.3 + 
                     flow_score * 0.2 + 
                     gap_score * 0.1) * vol_weight * elasticity_adjustment
        
        confluence_scores.append(confluence)
    
    df['confluence_score'] = confluence_scores
    
    # Dynamic risk assessment and final factor
    risk_scores = []
    for i in range(len(df)):
        if i < 20:
            risk_scores.append(0)
            continue
            
        current = df.iloc[i]
        
        # Risk assessment components
        regime_mismatch = 0
        if (current['vol_regime_class'] == 'high_vol' and 
            current['pressure_regime_class'] == 'high_pressure'):
            regime_mismatch = -0.3  # High volatility + high pressure = noise
        
        flow_breakdown = 0
        if current['flow_synchronization'] < 0:
            flow_breakdown = -0.2
            
        gap_risk = 0
        if (abs(current['gap_size']) > 0 and 
            abs(current['gap_fill_efficiency']) < 0.3):
            gap_risk = -0.2
            
        # Adaptive confidence
        confidence = 1.0 + regime_mismatch + flow_breakdown + gap_risk
        
        risk_scores.append(confidence)
    
    df['risk_adjustment'] = risk_scores
    
    # Final factor with risk adjustment
    result = df['confluence_score'] * df['risk_adjustment']
    
    # Clean up intermediate columns
    result = result.fillna(0)
    
    return result
