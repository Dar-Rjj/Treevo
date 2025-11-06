import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum Asymmetry Analysis
    data['upside_momentum'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    data['downside_momentum'] = (data['high'] - data['close']) / (data['high'] - data['low'] + 1e-8)
    data['momentum_skewness'] = data['upside_momentum'] - data['downside_momentum']
    
    # Calculate momentum persistence
    data['price_change'] = data['close'] - data['close'].shift(1)
    data['price_change_sign'] = np.sign(data['price_change'])
    
    momentum_persistence = []
    for i in range(len(data)):
        if i < 5:
            momentum_persistence.append(0)
        else:
            window = data['price_change_sign'].iloc[i-4:i+1]
            if len(window) == 0:
                momentum_persistence.append(0)
            else:
                current_sign = window.iloc[-1]
                if current_sign == 0:
                    momentum_persistence.append(0)
                else:
                    count = 0
                    for j in range(len(window)-1, -1, -1):
                        if window.iloc[j] == current_sign:
                            count += 1
                        else:
                            break
                    momentum_persistence.append(count)
    data['momentum_persistence'] = momentum_persistence
    
    # Flow Imbalance Detection
    data['buy_pressure_flow'] = data['volume'] * data['upside_momentum']
    data['sell_pressure_flow'] = data['volume'] * data['downside_momentum']
    data['net_flow_imbalance'] = (data['buy_pressure_flow'] - data['sell_pressure_flow']) / (data['volume'] + 1e-8)
    data['flow_concentration_ratio'] = np.maximum(data['buy_pressure_flow'], data['sell_pressure_flow']) / (data['volume'] + 1e-8)
    
    # Price-Volume Asymmetry
    data['volume_weighted_price_position'] = data['upside_momentum'] * data['volume']
    
    price_change_abs = np.abs(data['price_change'])
    data['asymmetric_volume_efficiency'] = (data['buy_pressure_flow'] - data['sell_pressure_flow']) / (price_change_abs + 1e-8)
    
    # Calculate rolling correlations for price-flow divergence
    price_flow_divergence = []
    for i in range(len(data)):
        if i < 3:
            price_flow_divergence.append(0)
        else:
            window_prices = data['price_change'].iloc[i-2:i+1]
            window_flows = data['net_flow_imbalance'].iloc[i-2:i+1]
            if len(window_prices) >= 2 and len(window_flows) >= 2:
                corr = window_prices.corr(window_flows)
                price_flow_divergence.append(corr if not np.isnan(corr) else 0)
            else:
                price_flow_divergence.append(0)
    data['price_flow_divergence'] = price_flow_divergence
    
    # Flow momentum asymmetry
    data['buy_flow_3d_ratio'] = data['buy_pressure_flow'] / (data['buy_pressure_flow'].shift(3) + 1e-8)
    data['sell_flow_3d_ratio'] = data['sell_pressure_flow'] / (data['sell_pressure_flow'].shift(3) + 1e-8)
    data['flow_momentum_asymmetry'] = data['buy_flow_3d_ratio'] - data['sell_flow_3d_ratio']
    
    # Regime-Dependent Signals
    data['volume_5d_ma'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_ratio'] = data['volume'] / (data['volume_5d_ma'] + 1e-8)
    data['high_flow_regime'] = (data['volume_ratio'] > 1.2).astype(int)
    data['low_flow_regime'] = (data['volume_ratio'] < 0.8).astype(int)
    
    data['regime_adaptive_momentum'] = data['momentum_skewness'] * (1 + data['high_flow_regime']) * (1 - data['low_flow_regime'])
    data['regime_specific_flow'] = data['net_flow_imbalance'] * (1 + data['high_flow_regime'] - data['low_flow_regime'])
    
    # Microstructure Quality Assessment
    data['price_efficiency_score'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # Calculate flow consistency measure
    flow_consistency = []
    for i in range(len(data)):
        if i < 3:
            flow_consistency.append(0)
        else:
            window_buy = data['buy_pressure_flow'].iloc[i-2:i+1]
            window_sell = data['sell_pressure_flow'].iloc[i-2:i+1]
            if len(window_buy) >= 2 and len(window_sell) >= 2:
                corr = window_buy.corr(window_sell)
                flow_consistency.append(corr if not np.isnan(corr) else 0)
            else:
                flow_consistency.append(0)
    data['flow_consistency_measure'] = flow_consistency
    
    # Asymmetry persistence
    asymmetry_persistence = []
    for i in range(len(data)):
        if i < 5:
            asymmetry_persistence.append(0)
        else:
            window = data['momentum_skewness'].iloc[i-4:i+1]
            if len(window) == 0:
                asymmetry_persistence.append(0)
            else:
                current_sign = np.sign(window.iloc[-1])
                if current_sign == 0:
                    asymmetry_persistence.append(0)
                else:
                    count = 0
                    for j in range(len(window)-1, -1, -1):
                        if np.sign(window.iloc[j]) == current_sign:
                            count += 1
                        else:
                            break
                    asymmetry_persistence.append(count)
    data['asymmetry_persistence'] = asymmetry_persistence
    
    data['quality_adjusted_flow'] = data['net_flow_imbalance'] * data['price_efficiency_score'] * (1 - np.abs(data['flow_consistency_measure']))
    
    # Composite Signal Construction
    data['core_asymmetry_component'] = data['momentum_skewness'] * data['net_flow_imbalance'] * data['flow_momentum_asymmetry']
    data['regime_enhanced_signal'] = data['core_asymmetry_component'] * data['regime_adaptive_momentum'] * data['regime_specific_flow']
    data['quality_filtered_asymmetry'] = data['regime_enhanced_signal'] * data['quality_adjusted_flow'] * data['asymmetry_persistence']
    data['volume_confirmed_prediction'] = data['quality_filtered_asymmetry'] * data['flow_concentration_ratio'] * data['momentum_persistence']
    
    # Final Alpha Generation
    data['amount_5d_ma'] = data['amount'].rolling(window=5, min_periods=1).mean()
    data['amount_ratio'] = data['amount'] / (data['amount_5d_ma'] + 1e-8)
    
    data['asymmetry_momentum_factor'] = data['volume_confirmed_prediction'] * data['amount_ratio']
    data['flow_regime_factor'] = data['asymmetry_momentum_factor'] * (1 + data['high_flow_regime'] - data['low_flow_regime'])
    
    # Final alpha
    final_alpha = data['flow_regime_factor'] * data['price_flow_divergence'] * (1 - np.abs(data['flow_consistency_measure']))
    
    return final_alpha
