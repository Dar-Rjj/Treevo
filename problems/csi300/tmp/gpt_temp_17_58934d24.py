import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # 1. Microstructure Entropy Components
    # Intraday Pressure
    data['intraday_pressure'] = (data['high'] - data['open']) - (data['open'] - data['low'])
    
    # Movement Efficiency
    high_low_range = data['high'] - data['low']
    high_prev_close = abs(data['high'] - data['close'].shift(1))
    low_prev_close = abs(data['low'] - data['close'].shift(1))
    denominator = pd.concat([high_low_range, high_prev_close, low_prev_close], axis=1).max(axis=1)
    data['movement_efficiency'] = abs(data['close'] - data['close'].shift(1)) / denominator.replace(0, np.nan)
    
    # Efficiency-Weighted Imbalance
    data['efficiency_weighted_imbalance'] = data['intraday_pressure'] * data['movement_efficiency']
    
    # 5-day and 10-day efficiency entropy
    def calculate_entropy(series, window):
        entropy_values = []
        for i in range(len(series)):
            if i >= window - 1:
                window_data = series.iloc[i-window+1:i+1].abs()
                total_sum = window_data.sum()
                if total_sum > 0:
                    probabilities = window_data / total_sum
                    entropy = -(probabilities * np.log(probabilities.replace(0, 1e-10))).sum()
                else:
                    entropy = 0
            else:
                entropy = np.nan
            entropy_values.append(entropy)
        return pd.Series(entropy_values, index=series.index)
    
    data['entropy_5d'] = calculate_entropy(data['efficiency_weighted_imbalance'], 5)
    data['entropy_10d'] = calculate_entropy(data['efficiency_weighted_imbalance'], 10)
    
    # 2. Fractal Momentum Patterns
    # 3-day efficiency momentum
    data['momentum_3d'] = (data['efficiency_weighted_imbalance'] / data['efficiency_weighted_imbalance'].shift(2)) * \
                         (data['efficiency_weighted_imbalance'].shift(1) / data['efficiency_weighted_imbalance'].shift(3))
    
    # 5-day efficiency momentum  
    data['momentum_5d'] = (data['efficiency_weighted_imbalance'] / data['efficiency_weighted_imbalance'].shift(4)) * \
                         (data['efficiency_weighted_imbalance'].shift(2) / data['efficiency_weighted_imbalance'].shift(6))
    
    # Momentum consistency
    data['momentum_consistency'] = ((data['momentum_3d'] > 0) & (data['momentum_5d'] > 0)).astype(int) + \
                                  ((data['momentum_3d'] < 0) & (data['momentum_5d'] < 0)).astype(int)
    
    # Entropy-Momentum Quality
    data['entropy_divergence'] = (data['entropy_10d'] - data['entropy_5d']) * data['momentum_consistency']
    avg_momentum = (data['momentum_3d'] + data['momentum_5d']) / 2
    data['entropy_momentum_quality'] = np.tanh(data['entropy_divergence'] * avg_momentum)
    
    # 3. Volatility-Velocity Alignment
    # Microstructure Volatility Ratio
    sum_5d = data['efficiency_weighted_imbalance'].abs().rolling(window=5).sum()
    sum_10d = data['efficiency_weighted_imbalance'].abs().rolling(window=10).sum()
    data['microstructure_vol_ratio'] = (sum_5d / sum_10d) - 1
    
    # Volatility Regime
    realized_vol_5d = data['close'].pct_change().rolling(window=5).std()
    realized_vol_10d = data['close'].pct_change().rolling(window=10).std()
    data['volatility_regime'] = realized_vol_5d / realized_vol_10d
    data['regime_indicator'] = ((data['volatility_regime'] > 1.5) | (data['volatility_regime'] < 0.7)).astype(int)
    
    # Velocity Components
    data['price_velocity'] = (data['close'] - data['close'].shift(2)) / (data['close'].shift(2) - data['close'].shift(4)).replace(0, np.nan)
    data['efficiency_velocity'] = data['movement_efficiency'] / data['movement_efficiency'].shift(2).replace(0, np.nan)
    data['pressure_velocity'] = data['intraday_pressure'] / data['intraday_pressure'].shift(2).replace(0, np.nan)
    
    # Volatility-Velocity Confluence
    data['velocity_alignment'] = np.sign(data['price_velocity']) * np.sign(data['efficiency_velocity']) * \
                               np.minimum(abs(data['price_velocity']), abs(data['efficiency_velocity']))
    data['vol_velocity_confluence'] = data['velocity_alignment'] * data['pressure_velocity'] * \
                                     data['microstructure_vol_ratio'] * data['regime_indicator']
    
    # 4. Liquidity-Discovery Synthesis
    # Discovery Components
    data['intraday_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['discovery_momentum'] = (data['close'] - data['close'].shift(5)) * \
                                (data['intraday_efficiency'] + data['movement_efficiency']) * \
                                np.sign(data['intraday_pressure'])
    
    # Correlation for Discovery Quality
    def rolling_corr(series1, series2, window):
        corr_values = []
        for i in range(len(series1)):
            if i >= window - 1:
                window1 = series1.iloc[i-window+1:i+1]
                window2 = series2.iloc[i-window+1:i+1]
                if len(window1) == window and len(window2) == window:
                    corr = window1.corr(window2)
                else:
                    corr = np.nan
            else:
                corr = np.nan
            corr_values.append(corr)
        return pd.Series(corr_values, index=series1.index)
    
    data['discovery_correlation'] = rolling_corr(data['intraday_efficiency'], data['movement_efficiency'], 5)
    data['discovery_quality'] = data['discovery_momentum'] * data['discovery_correlation']
    
    # Liquidity Components
    avg_amount_5d = data['amount'].rolling(window=5).mean()
    data['amount_weighted_pressure'] = data['intraday_pressure'] * (data['amount'] / avg_amount_5d.replace(0, np.nan))
    data['liquidity_efficiency'] = abs(data['close'] - data['close'].shift(1)) / data['volume'].replace(0, np.nan)
    
    numerator = abs((data['close'] - data['open']) * data['volume'])
    denominator = abs((data['open'] - data['close'].shift(1)) * data['volume']) + numerator
    data['flow_asymmetry'] = numerator / denominator.replace(0, np.nan)
    
    data['informed_flow'] = data['amount_weighted_pressure'] * data['liquidity_efficiency'] * data['flow_asymmetry']
    
    # Price-Volume correlation
    data['price_volume_correlation'] = rolling_corr(data['close'], data['volume'], 5)
    data['liquidity_discovery_quality'] = data['discovery_quality'] * data['informed_flow'] * data['price_volume_correlation']
    
    # 5. Adaptive Multi-Fractal Synthesis
    # Component Integration
    integrated_product = data['entropy_momentum_quality'] * data['vol_velocity_confluence'] * data['liquidity_discovery_quality']
    data['integrated_score'] = np.sign(integrated_product) * np.power(abs(integrated_product), 1/3)
    
    # Signal Validation
    momentum_filter = data['momentum_consistency'] >= 2
    velocity_filter = (np.sign(data['price_velocity']) == np.sign(data['efficiency_velocity']))
    volatility_filter = data['regime_indicator'] == 1
    
    # Final Alpha
    data['alpha'] = data['integrated_score'] * momentum_filter * velocity_filter * volatility_filter * \
                   np.sign(data['intraday_pressure']) * data['movement_efficiency']
    
    return data['alpha']
