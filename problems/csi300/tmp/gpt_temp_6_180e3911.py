import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Volatility Regime Assessment
    # True Range
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Average True Range (5-day rolling)
    data['atr_5'] = data['true_range'].rolling(window=5, min_periods=1).mean().shift(1)
    data['volatility_ratio'] = data['true_range'] / data['atr_5']
    
    # Price Efficiency Asymmetry
    # Up Efficiency
    up_mask = data['close'] > data['open']
    data['up_efficiency'] = 0.0
    data.loc[up_mask, 'up_efficiency'] = (data.loc[up_mask, 'close'] - data.loc[up_mask, 'open']) / (data.loc[up_mask, 'high'] - data.loc[up_mask, 'open'])
    
    # Down Efficiency
    down_mask = data['close'] < data['open']
    data['down_efficiency'] = 0.0
    data.loc[down_mask, 'down_efficiency'] = (data.loc[down_mask, 'open'] - data.loc[down_mask, 'close']) / (data.loc[down_mask, 'open'] - data.loc[down_mask, 'low'])
    
    # Efficiency Asymmetry
    data['efficiency_asymmetry'] = data['up_efficiency'] - data['down_efficiency']
    
    # Regime-Aware Momentum
    # Short-term Momentum
    data['short_momentum'] = data['close'] / data['close'].shift(5) - 1
    
    # Pattern Correlation
    def calculate_pattern_correlation(close_series, window=20):
        correlations = []
        for i in range(len(close_series)):
            if i < window + 4:
                correlations.append(0.0)
                continue
                
            current_pattern = close_series.iloc[i-4:i+1].pct_change().dropna().values
            if len(current_pattern) != 4:
                correlations.append(0.0)
                continue
                
            corr_values = []
            for j in range(window):
                if i - j - 5 < 0:
                    continue
                    
                past_pattern = close_series.iloc[i-j-5:i-j].pct_change().dropna().values
                if len(past_pattern) == 4:
                    corr = np.corrcoef(current_pattern, past_pattern)[0, 1]
                    if not np.isnan(corr):
                        corr_values.append(corr)
            
            if corr_values:
                correlations.append(np.mean(corr_values))
            else:
                correlations.append(0.0)
        
        return pd.Series(correlations, index=close_series.index)
    
    data['pattern_corr'] = calculate_pattern_correlation(data['close'])
    
    # Volume Transition Signal
    # Volume Surge
    data['avg_volume_5'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_surge'] = data['volume'] / data['avg_volume_5']
    
    # Volume Clustering
    data['volume_clustering'] = -data['volume_surge'] * np.log(data['volume_surge'].clip(lower=1e-6))
    
    # Alpha Construction
    # Core Factor
    data['core_factor'] = data['efficiency_asymmetry'] * (data['short_momentum'] * data['pattern_corr'])
    
    # Volume Adjusted
    data['volume_adjusted'] = data['core_factor'] * (data['volume_surge'] * data['volume_clustering'])
    
    # Final Alpha
    data['alpha'] = data['volume_adjusted'] * data['volatility_ratio'] * data['true_range']
    
    return data['alpha']
