import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Cross-Asset Momentum Components
    data['price_momentum'] = data['close'] / data['close'].shift(1) - 1
    data['volume_momentum'] = data['volume'] / data['volume'].shift(1) - 1
    data['volatility_momentum'] = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1)) - 1
    
    # Handle division by zero in volatility momentum
    data['volatility_momentum'] = data['volatility_momentum'].replace([np.inf, -np.inf], np.nan)
    
    # Microstructure Divergence Analysis
    data['mid_range_anchor'] = (data['high'] + data['low']) / 2
    data['close_to_mid_range_deviation'] = data['close'] / data['mid_range_anchor'] - 1
    
    # Efficiency asymmetry calculation
    high_open_diff = data['high'] - data['open']
    open_low_diff = data['open'] - data['low']
    
    # Handle division by zero cases
    high_open_diff = high_open_diff.replace(0, np.nan)
    open_low_diff = open_low_diff.replace(0, np.nan)
    
    data['efficiency_asymmetry'] = ((data['close'] - data['open']) / high_open_diff) - \
                                  ((data['open'] - data['close']) / open_low_diff)
    
    # Price-volume divergence (5-day correlation)
    data['price_volume_divergence'] = data['price_momentum'].rolling(window=5).corr(data['volume_momentum'])
    
    # Volatility Regime Integration
    data['vol_10d'] = data['close'].rolling(window=10).std()
    data['vol_30d'] = data['close'].rolling(window=30).std()
    data['volatility_ratio'] = data['vol_10d'] / data['vol_30d']
    data['volatility_expansion'] = (data['volatility_ratio'] > 1.2).astype(int)
    
    # Volatility momentum persistence (consecutive same-sign changes)
    vol_momentum_sign = np.sign(data['volatility_momentum'])
    vol_persistence = []
    current_streak = 0
    
    for i in range(len(vol_momentum_sign)):
        if i == 0 or pd.isna(vol_momentum_sign.iloc[i]) or pd.isna(vol_momentum_sign.iloc[i-1]):
            current_streak = 0
        elif vol_momentum_sign.iloc[i] == vol_momentum_sign.iloc[i-1] and vol_momentum_sign.iloc[i] != 0:
            current_streak += 1
        else:
            current_streak = 0
        vol_persistence.append(current_streak)
    
    data['volatility_momentum_persistence'] = vol_persistence
    
    # Momentum Persistence Patterns
    # Price streak count
    price_change_sign = np.sign(data['price_momentum'])
    price_streak = []
    current_streak = 0
    
    for i in range(len(price_change_sign)):
        if i == 0 or pd.isna(price_change_sign.iloc[i]) or pd.isna(price_change_sign.iloc[i-1]):
            current_streak = 0
        elif price_change_sign.iloc[i] == price_change_sign.iloc[i-1] and price_change_sign.iloc[i] != 0:
            current_streak += 1
        else:
            current_streak = 0
        price_streak.append(current_streak)
    
    data['price_streak_count'] = price_streak
    
    # Volume streak count
    volume_change_sign = np.sign(data['volume_momentum'])
    volume_streak = []
    current_streak = 0
    
    for i in range(len(volume_change_sign)):
        if i == 0 or pd.isna(volume_change_sign.iloc[i]) or pd.isna(volume_change_sign.iloc[i-1]):
            current_streak = 0
        elif volume_change_sign.iloc[i] == volume_change_sign.iloc[i-1] and volume_change_sign.iloc[i] != 0:
            current_streak += 1
        else:
            current_streak = 0
        volume_streak.append(current_streak)
    
    data['volume_streak_count'] = volume_streak
    
    # Volatility streak count
    volatility_streak = []
    current_streak = 0
    
    for i in range(len(vol_momentum_sign)):
        if i == 0 or pd.isna(vol_momentum_sign.iloc[i]) or pd.isna(vol_momentum_sign.iloc[i-1]):
            current_streak = 0
        elif vol_momentum_sign.iloc[i] == vol_momentum_sign.iloc[i-1] and vol_momentum_sign.iloc[i] != 0:
            current_streak += 1
        else:
            current_streak = 0
        volatility_streak.append(current_streak)
    
    data['volatility_streak_count'] = volatility_streak
    
    # Factor Synthesis
    data['momentum_anchor_convergence'] = data['price_momentum'] * data['close_to_mid_range_deviation']
    data['volatility_momentum_divergence'] = data['volatility_momentum'] * data['price_volume_divergence']
    data['efficiency_volatility_asymmetry'] = data['efficiency_asymmetry'] * data['volatility_ratio']
    data['persistence_weighted_volatility_factor'] = (data['price_streak_count'] + data['volatility_streak_count']) * data['volatility_momentum']
    
    # Final alpha factor
    data['regime_confirmed_alpha'] = data['volatility_expansion'] * (
        data['momentum_anchor_convergence'] + 
        data['volatility_momentum_divergence']
    )
    
    # Return the final alpha factor series
    return data['regime_confirmed_alpha']
