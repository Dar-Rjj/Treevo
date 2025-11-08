import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate True Range
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Volatility-Weighted Price Asymmetry
    data['upside_vol'] = np.maximum(0, data['high'] - data['prev_close'])
    data['downside_vol'] = np.maximum(0, data['prev_close'] - data['low'])
    
    # Handle zero cases for asymmetry ratio
    mask = (data['downside_vol'] > 0) & (data['upside_vol'] > 0)
    data['asymmetry_ratio'] = 0.0
    data.loc[mask, 'asymmetry_ratio'] = np.log(data.loc[mask, 'upside_vol'] / data.loc[mask, 'downside_vol'])
    
    # Volume-Momentum Divergence
    data['volume_ratio_t'] = data['volume'] / data['volume'].shift(1)
    data['volume_ratio_t1'] = data['volume'].shift(1) / data['volume'].shift(2)
    data['volume_acceleration'] = data['volume_ratio_t'] - data['volume_ratio_t1']
    
    # Volatility-Adjusted Momentum
    data['momentum_5d'] = (data['close'] / data['close'].shift(5)) - 1
    data['vol_adj_momentum'] = data['momentum_5d'] / data['true_range']
    
    # Divergence
    data['divergence'] = data['asymmetry_ratio'] * data['volume_acceleration']
    
    # Market Microstructure Integration
    data['efficiency_indicator'] = abs(data['close'] - data['open']) / (data['high'] - data['low'])
    data['efficiency_indicator'] = data['efficiency_indicator'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Volatility Ratio using rolling standard deviation
    data['std_10'] = data['close'].rolling(window=10, min_periods=5).std()
    data['std_20'] = data['close'].rolling(window=20, min_periods=10).std()
    data['volatility_ratio'] = data['std_10'] / data['std_20']
    data['volatility_ratio'] = data['volatility_ratio'].replace([np.inf, -np.inf], 1.0).fillna(1.0)
    
    # Persistence Analysis
    def calculate_persistence(series):
        persistence = pd.Series(0, index=series.index)
        current_streak = 0
        prev_sign = 0
        
        for i in range(len(series)):
            if pd.isna(series.iloc[i]):
                current_streak = 0
                prev_sign = 0
                persistence.iloc[i] = 0
                continue
                
            current_sign = 1 if series.iloc[i] > 0 else (-1 if series.iloc[i] < 0 else 0)
            
            if current_sign == prev_sign and current_sign != 0:
                current_streak += 1
            else:
                current_streak = 1 if current_sign != 0 else 0
                
            persistence.iloc[i] = current_streak
            prev_sign = current_sign
            
        return persistence
    
    data['asymmetry_persistence'] = calculate_persistence(data['asymmetry_ratio'])
    data['divergence_persistence'] = calculate_persistence(data['divergence'])
    
    # Final Alpha Construction
    # Core Factor
    data['core_factor'] = data['asymmetry_ratio'] * data['volume_acceleration'] * data['vol_adj_momentum']
    
    # Persistence Adjustment
    data['persistence_adjustment'] = data['core_factor'] * (1 + 0.05 * data['asymmetry_persistence'])
    
    # Efficiency Weighting
    data['efficiency_weighted'] = data['persistence_adjustment'] * data['efficiency_indicator']
    
    # Volatility Scaling
    volatility_scale = np.where(data['volatility_ratio'] > 1.0, data['volatility_ratio'], 0.7)
    data['final_alpha'] = data['efficiency_weighted'] * volatility_scale
    
    # Clean up and return
    result = data['final_alpha'].replace([np.inf, -np.inf], 0).fillna(0)
    return result
