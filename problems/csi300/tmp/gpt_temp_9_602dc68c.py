import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Calculate returns and volume changes
    data['ret'] = data['close'] / data['close'].shift(1) - 1
    data['vol_change'] = data['volume'] / data['volume'].shift(1) - 1
    
    # Multi-Timeframe Momentum Entropy
    def calculate_entropy(returns, window):
        abs_returns = returns.abs()
        rolling_sum = abs_returns.rolling(window=window).sum()
        probabilities = abs_returns / rolling_sum
        entropy = -(probabilities * np.log(probabilities + 1e-8)).rolling(window=window).sum()
        return entropy
    
    # Momentum entropy components
    data['momentum_entropy_5'] = calculate_entropy(data['ret'], 5)
    data['momentum_entropy_10'] = calculate_entropy(data['ret'], 10)
    data['momentum_entropy_gradient'] = data['momentum_entropy_10'] - data['momentum_entropy_5']
    
    # Volume entropy components
    def calculate_volume_entropy(volume, window):
        rolling_sum = volume.rolling(window=window).sum()
        probabilities = volume / rolling_sum
        entropy = -(probabilities * np.log(probabilities + 1e-8)).rolling(window=window).sum()
        return entropy
    
    data['volume_entropy_5'] = calculate_volume_entropy(data['volume'], 5)
    data['volume_entropy_10'] = calculate_volume_entropy(data['volume'], 10)
    data['volume_entropy_gradient'] = data['volume_entropy_10'] - data['volume_entropy_5']
    
    # Combined entropy signal
    data['combined_entropy_signal'] = data['momentum_entropy_gradient'] * data['volume_entropy_gradient']
    
    # Asymmetric Momentum Fractals
    # Short-term momentum fractal
    data['short_fractal'] = (data['close'] / data['close'].shift(2) - 1) * (data['close'].shift(1) / data['close'].shift(3) - 1)
    
    # Medium-term momentum fractal
    data['medium_fractal'] = (data['close'] / data['close'].shift(4) - 1) * (data['close'].shift(2) / data['close'].shift(6) - 1)
    
    # Fractal consistency
    data['fractal_consistency'] = ((data['short_fractal'] > 0).astype(int) + 
                                  (data['medium_fractal'] > 0).astype(int))
    
    # Asymmetric Volatility Integration
    data['upside_vol_efficiency'] = ((data['high'] - data['open']) / data['open']) * (data['volume'] / data['volume'].shift(1))
    data['downside_vol_efficiency'] = ((data['open'] - data['low']) / data['open']) * (data['volume'] / data['volume'].shift(1))
    data['volatility_asymmetry_ratio'] = data['upside_vol_efficiency'] / (data['downside_vol_efficiency'] + 1e-8)
    
    # Liquidity Flow Dynamics
    data['opening_momentum_flow'] = (data['open'] - data['close'].shift(1)) * data['volume']
    data['closing_momentum_flow'] = (data['close'] - data['open']) * data['volume']
    data['flow_momentum_ratio'] = (data['closing_momentum_flow'].abs() / 
                                  (data['opening_momentum_flow'].abs() + data['closing_momentum_flow'].abs() + 1e-8))
    
    # Volume-Price Efficiency
    data['current_price_efficiency'] = data['ret'].abs() / data['volume']
    data['avg_efficiency_5'] = (data['ret'].abs() / data['volume']).rolling(window=5).mean()
    data['efficiency_ratio'] = data['current_price_efficiency'] / (data['avg_efficiency_5'] + 1e-8)
    
    # Cross-Scale Correlation Structure
    def calculate_correlation(data_col1, data_col2, window):
        corrs = []
        for i in range(len(data)):
            if i >= window - 1:
                window_data1 = data_col1.iloc[i-window+1:i+1]
                window_data2 = data_col2.iloc[i-window+1:i+1]
                if len(window_data1) == window and len(window_data2) == window:
                    corr = window_data1.corr(window_data2)
                    corrs.append(corr if not np.isnan(corr) else 0)
                else:
                    corrs.append(0)
            else:
                corrs.append(0)
        return pd.Series(corrs, index=data.index)
    
    data['corr_3d'] = calculate_correlation(data['ret'], data['vol_change'], 3)
    data['corr_8d'] = calculate_correlation(data['ret'], data['vol_change'], 8)
    data['correlation_gradient'] = data['corr_8d'] - data['corr_3d']
    
    # Lead-Lag Momentum Patterns
    def calculate_lead_lag_correlation(lead_col, lag_col, window):
        corrs = []
        for i in range(len(data)):
            if i >= window:
                lead_window = lead_col.iloc[i-window:i]
                lag_window = lag_col.iloc[i-window+1:i+1]
                if len(lead_window) == window and len(lag_window) == window:
                    corr = lead_window.corr(lag_window)
                    corrs.append(corr if not np.isnan(corr) else 0)
                else:
                    corrs.append(0)
            else:
                corrs.append(0)
        return pd.Series(corrs, index=data.index)
    
    data['volume_leading_momentum'] = calculate_lead_lag_correlation(data['vol_change'], data['ret'], 7)
    data['momentum_leading_volume'] = calculate_lead_lag_correlation(data['ret'], data['vol_change'], 7)
    data['lead_lag_dominance'] = data['volume_leading_momentum'] - data['momentum_leading_volume']
    
    # Adaptive Signal Synthesis
    # Core Entropy-Momentum Integration
    data['base_entropy_signal'] = data['combined_entropy_signal'] * data['fractal_consistency']
    data['volatility_enhanced_entropy'] = data['base_entropy_signal'] * data['volatility_asymmetry_ratio']
    data['flow_weighted_entropy'] = data['volatility_enhanced_entropy'] * data['flow_momentum_ratio']
    
    # Efficiency-Based Refinement
    data['correlation_adjustment'] = data['flow_weighted_entropy'] * data['correlation_gradient']
    data['lead_lag_enhancement'] = data['correlation_adjustment'] * data['lead_lag_dominance']
    data['final_efficiency_filter'] = data['lead_lag_enhancement'] * data['efficiency_ratio']
    
    # Alpha Generation
    data['raw_alpha_signal'] = data['final_efficiency_filter'] * data['volume_entropy_gradient']
    data['refined_alpha'] = data['raw_alpha_signal'] * data['momentum_entropy_gradient']
    
    return data['refined_alpha']
