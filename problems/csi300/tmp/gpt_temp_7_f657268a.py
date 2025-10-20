import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining multi-timeframe momentum, volume confirmation,
    directional convergence, and price range analysis.
    """
    # Create copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Price Momentum Analysis
    data['M5'] = data['close'] / data['close'].shift(5) - 1
    data['M20'] = data['close'] / data['close'].shift(20) - 1
    data['M60'] = data['close'] / data['close'].shift(60) - 1
    data['M120'] = data['close'] / data['close'].shift(120) - 1
    
    # Weighted momentum base
    weights = {'M5': 0.4, 'M20': 0.3, 'M60': 0.2, 'M120': 0.1}
    data['WM'] = (weights['M5'] * data['M5'] + 
                  weights['M20'] * data['M20'] + 
                  weights['M60'] * data['M60'] + 
                  weights['M120'] * data['M120'])
    
    # Volume Confirmation Framework
    data['VR5'] = data['volume'] / data['volume'].shift(5)
    data['VR20'] = data['volume'] / data['volume'].shift(20)
    data['VR60'] = data['volume'] / data['volume'].shift(60)
    data['VR120'] = data['volume'] / data['volume'].shift(120)
    
    # Volume confirmation scoring
    volume_scores = []
    for i, row in data.iterrows():
        score = 0
        # Check alignment for each timeframe
        if not pd.isna(row['M5']) and not pd.isna(row['VR5']):
            if (row['M5'] > 0 and row['VR5'] > 1) or (row['M5'] < 0 and row['VR5'] < 1):
                score += 1
        if not pd.isna(row['M20']) and not pd.isna(row['VR20']):
            if (row['M20'] > 0 and row['VR20'] > 1) or (row['M20'] < 0 and row['VR20'] < 1):
                score += 1
        if not pd.isna(row['M60']) and not pd.isna(row['VR60']):
            if (row['M60'] > 0 and row['VR60'] > 1) or (row['M60'] < 0 and row['VR60'] < 1):
                score += 1
        if not pd.isna(row['M120']) and not pd.isna(row['VR120']):
            if (row['M120'] > 0 and row['VR120'] > 1) or (row['M120'] < 0 and row['VR120'] < 1):
                score += 1
        volume_scores.append(score)
    
    data['volume_score'] = volume_scores
    
    # Volume confirmation multiplier
    def get_volume_multiplier(score):
        if score == 4:
            return 2.0
        elif score == 3:
            return 1.7
        elif score == 2:
            return 1.3
        elif score == 1:
            return 1.0
        else:
            return 0.5
    
    data['volume_multiplier'] = data['volume_score'].apply(get_volume_multiplier)
    
    # Directional Convergence Assessment
    def get_directional_convergence(row):
        if pd.isna(row['M5']) or pd.isna(row['M20']) or pd.isna(row['M60']) or pd.isna(row['M120']):
            return 0.4  # Divergence for missing data
        
        positives = sum([1 for m in [row['M5'], row['M20'], row['M60'], row['M120']] if m > 0])
        negatives = sum([1 for m in [row['M5'], row['M20'], row['M60'], row['M120']] if m < 0])
        
        if positives == 4 or negatives == 4:
            return 1.6  # Strong convergence
        elif positives == 3 or negatives == 3:
            return 1.3  # High convergence
        elif positives == 2 or negatives == 2:
            return 1.0  # Moderate convergence
        else:
            return 0.4  # Divergence
    
    data['direction_multiplier'] = data.apply(get_directional_convergence, axis=1)
    
    # Price Range Analysis
    data['range_eff'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    
    # Calculate previous day's range efficiency
    prev_range_eff = (data['close'].shift(1) - data['low'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1))
    data['range_consistency'] = data['range_eff'] - prev_range_eff
    
    # Range multiplier
    data['range_multiplier'] = 0.5 + data['range_eff'] + 0.5 * data['range_consistency']
    
    # Final alpha factor calculation
    data['alpha'] = (data['WM'] * 
                     data['volume_multiplier'] * 
                     data['direction_multiplier'] * 
                     data['range_multiplier'])
    
    return data['alpha']
