import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic price-based features
    data['returns'] = data['close'].pct_change()
    data['spread_est'] = (data['high'] - data['low']) / data['close']  # Proxy for bid-ask spread
    
    # Bid-Ask Spread Dynamics
    data['relative_spread_change'] = data['spread_est'].pct_change()
    data['spread_volatility'] = data['spread_est'].rolling(window=5, min_periods=3).std()
    data['spread_momentum_factor'] = data['relative_spread_change'] / (data['spread_volatility'] + 1e-8)
    
    data['spread_compression_ratio'] = data['volume'] / (data['spread_est'] + 1e-8)
    data['spread_expansion_pressure'] = (data['high'] - data['low']) / (data['spread_est'] + 1e-8)
    data['spread_efficiency'] = data['spread_compression_ratio'] * data['spread_expansion_pressure']
    
    # Market Maker Behavior (using volume and price as proxies)
    data['quote_intensity'] = data['volume'].rolling(window=5, min_periods=3).mean() / (data['volume'] + 1e-8)
    mid_price = (data['high'] + data['low']) / 2
    data['market_maker_stress'] = abs(data['close'] - mid_price) / (data['spread_est'] + 1e-8)
    data['market_maker_confidence'] = data['quote_intensity'] / (1 + data['market_maker_stress'])
    
    # Order Flow Imbalance Analysis (using price and volume patterns)
    price_change = data['close'].diff()
    data['buy_sell_pressure'] = np.where(price_change > 0, data['volume'], 
                                        np.where(price_change < 0, -data['volume'], 0)) / (data['volume'] + 1e-8)
    
    # Calculate rolling correlation for imbalance persistence
    imbalance_persistence = []
    for i in range(len(data)):
        if i >= 3:
            window_data = data.iloc[i-2:i+1]
            if len(window_data) >= 2:
                corr = window_data['buy_sell_pressure'].corr(window_data['returns'])
                imbalance_persistence.append(corr if not np.isnan(corr) else 0)
            else:
                imbalance_persistence.append(0)
        else:
            imbalance_persistence.append(0)
    data['imbalance_persistence'] = imbalance_persistence
    data['order_flow_momentum'] = data['buy_sell_pressure'] * data['imbalance_persistence']
    
    # Large Order Detection (using volume spikes)
    volume_ma = data['volume'].rolling(window=10, min_periods=5).mean()
    volume_std = data['volume'].rolling(window=10, min_periods=5).std()
    data['block_trade_ratio'] = np.where(data['volume'] > volume_ma + volume_std, 
                                        (data['volume'] - volume_ma) / (volume_std + 1e-8), 0)
    
    # Calculate large order impact
    large_order_impact = []
    for i in range(len(data)):
        if i >= 1 and data['block_trade_ratio'].iloc[i] > 0:
            impact = (data['close'].iloc[i] - data['close'].iloc[i-1]) / (data['spread_est'].iloc[i] + 1e-8)
            large_order_impact.append(impact)
        else:
            large_order_impact.append(0)
    data['large_order_impact'] = large_order_impact
    data['smart_money_indicator'] = data['block_trade_ratio'] * data['large_order_impact']
    
    # Order Flow Clustering
    data['order_clustering_intensity'] = data['volume'].diff().abs() / (data['volume'] + 1e-8)
    trade_size_proxy = data['amount'] / (data['volume'] + 1e-8)  # Average trade size proxy
    data['order_size_distribution'] = trade_size_proxy.rolling(window=10, min_periods=5).std() / \
                                     (trade_size_proxy.rolling(window=10, min_periods=5).mean() + 1e-8)
    data['flow_concentration'] = data['order_clustering_intensity'] * (1 - data['order_size_distribution'])
    
    # Price Impact Asymmetry
    upside_move = data['high'] - data['open']
    downside_move = data['open'] - data['low']
    buy_volume_proxy = np.where(price_change > 0, data['volume'], data['volume'] * 0.5)
    sell_volume_proxy = np.where(price_change < 0, data['volume'], data['volume'] * 0.5)
    
    data['upside_impact'] = upside_move / (buy_volume_proxy + 1e-8)
    data['downside_impact'] = downside_move / (sell_volume_proxy + 1e-8)
    data['impact_asymmetry'] = data['upside_impact'] - data['downside_impact']
    
    data['buy_sensitivity'] = data['upside_impact'] / (buy_volume_proxy + 1e-8)
    data['sell_sensitivity'] = data['downside_impact'] / (sell_volume_proxy + 1e-8)
    data['sensitivity_differential'] = data['buy_sensitivity'] - data['sell_sensitivity']
    
    # Calculate impact persistence correlations
    impact_memory = []
    impact_reversal = []
    for i in range(len(data)):
        if i >= 5:
            window_5 = data.iloc[i-4:i+1]
            corr_5 = window_5['impact_asymmetry'].corr(window_5['returns'])
            impact_memory.append(corr_5 if not np.isnan(corr_5) else 0)
            
            if i >= 2:
                window_2 = data.iloc[i-1:i+1]
                if len(window_2) >= 2:
                    corr_2 = window_2['impact_asymmetry'].corr(window_2['returns'])
                    impact_reversal.append(-corr_2 if not np.isnan(corr_2) else 0)
                else:
                    impact_reversal.append(0)
            else:
                impact_reversal.append(0)
        else:
            impact_memory.append(0)
            impact_reversal.append(0)
    
    data['impact_memory'] = impact_memory
    data['impact_reversal'] = impact_reversal
    data['impact_regime'] = data['impact_memory'] * (1 - abs(data['impact_reversal']))
    
    # Liquidity Regime Classification
    spread_ma = data['spread_est'].rolling(window=10, min_periods=5).mean()
    data['tight_spread'] = data['spread_est'] < spread_ma
    data['wide_spread'] = data['spread_est'] > spread_ma
    data['spread_regime_strength'] = data['market_maker_confidence'] * data['spread_efficiency']
    
    data['aggressive_buying'] = data['buy_sell_pressure'] > 0.2
    data['aggressive_selling'] = data['buy_sell_pressure'] < -0.2
    data['flow_regime_intensity'] = data['order_flow_momentum'] * data['flow_concentration']
    
    impact_asymmetry_std = data['impact_asymmetry'].rolling(window=20, min_periods=10).std()
    data['high_impact_asymmetry'] = abs(data['impact_asymmetry']) > impact_asymmetry_std
    data['low_impact_asymmetry'] = abs(data['impact_asymmetry']) < impact_asymmetry_std
    data['impact_regime_power'] = data['impact_regime'] * data['sensitivity_differential']
    
    # Cross-Dimension Alpha Signals
    data['spread_compression_alpha'] = data['spread_efficiency'] * data['order_flow_momentum']
    data['spread_expansion_alpha'] = data['spread_momentum_factor'] * data['smart_money_indicator']
    data['spread_flow_balance'] = data['spread_compression_alpha'] - data['spread_expansion_alpha']
    
    data['flow_impact_alpha'] = data['order_flow_momentum'] * data['impact_asymmetry']
    data['smart_impact_alpha'] = data['smart_money_indicator'] * data['sensitivity_differential']
    data['flow_impact_convergence'] = data['flow_impact_alpha'] * data['smart_impact_alpha']
    
    # Regime-Transition Detection
    data['spread_efficiency_prev'] = data['spread_efficiency'].shift(1)
    data['order_flow_momentum_prev'] = data['order_flow_momentum'].shift(1)
    data['impact_asymmetry_prev'] = data['impact_asymmetry'].shift(1)
    
    data['spread_regime_change'] = np.sign(data['spread_efficiency']).fillna(0) - np.sign(data['spread_efficiency_prev']).fillna(0)
    data['flow_regime_change'] = np.sign(data['order_flow_momentum']).fillna(0) - np.sign(data['order_flow_momentum_prev']).fillna(0)
    data['impact_regime_change'] = np.sign(data['impact_asymmetry']).fillna(0) - np.sign(data['impact_asymmetry_prev']).fillna(0)
    
    data['spread_transition'] = data['spread_regime_change'] * data['spread_momentum_factor']
    data['flow_transition'] = data['flow_regime_change'] * data['flow_concentration']
    data['impact_transition'] = data['impact_regime_change'] * data['impact_regime']
    data['transition_alpha'] = data['spread_transition'] + data['flow_transition'] + data['impact_transition']
    
    # Adaptive Alpha Integration
    # Spread-Driven Alpha
    data['spread_driven_alpha'] = np.where(
        data['tight_spread'], data['spread_compression_alpha'],
        np.where(data['wide_spread'], data['spread_expansion_alpha'], data['spread_flow_balance'])
    )
    
    # Flow-Driven Alpha
    data['flow_driven_alpha'] = np.where(
        data['aggressive_buying'], data['flow_impact_alpha'],
        np.where(data['aggressive_selling'], data['smart_impact_alpha'], data['flow_impact_convergence'])
    )
    
    # Impact-Driven Alpha
    data['impact_driven_alpha'] = np.where(
        data['high_impact_asymmetry'], data['impact_regime'] * data['flow_impact_convergence'],
        np.where(data['low_impact_asymmetry'], data['impact_regime'] * data['spread_flow_balance'],
                data['impact_regime'] * data['flow_impact_alpha'])
    )
    
    # Cross-Regime Signal Blending
    data['primary_alpha'] = data['spread_driven_alpha'] * data['spread_regime_strength']
    data['secondary_alpha'] = data['flow_driven_alpha'] * data['flow_regime_intensity']
    data['tertiary_alpha'] = data['impact_driven_alpha'] * data['impact_regime_power']
    
    # Dynamic Weighting Scheme
    denominator = (abs(data['spread_regime_strength']) + 
                  abs(data['flow_regime_intensity']) + 
                  abs(data['impact_regime_power']) + 1e-8)
    
    data['spread_weight'] = abs(data['spread_regime_strength']) / denominator
    data['flow_weight'] = abs(data['flow_regime_intensity']) / denominator
    data['impact_weight'] = abs(data['impact_regime_power']) / denominator
    
    # Final Alpha Synthesis
    final_alpha = (data['primary_alpha'] * data['spread_weight'] +
                  data['secondary_alpha'] * data['flow_weight'] +
                  data['tertiary_alpha'] * data['impact_weight'] +
                  data['transition_alpha'] * (1 - (data['spread_weight'] + data['flow_weight'] + data['impact_weight'])))
    
    return final_alpha
