import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Microstructure Momentum with Cross-Asset Confirmation
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Volatility & Liquidity Framework
    # Volatility Regime Identification
    data['true_range'] = np.maximum(data['high'] - data['low'], 
                                   np.maximum(abs(data['high'] - data['close'].shift(1)), 
                                             abs(data['low'] - data['close'].shift(1))))
    data['atr_5'] = data['true_range'].rolling(window=5).mean()
    data['intraday_vol'] = (data['high'] - data['low']) / data['close']
    
    # Volatility regime classification
    vol_percentile_30 = data['intraday_vol'].rolling(window=20).apply(lambda x: np.percentile(x, 30), raw=True)
    vol_percentile_70 = data['intraday_vol'].rolling(window=20).apply(lambda x: np.percentile(x, 70), raw=True)
    data['vol_regime'] = np.where(data['intraday_vol'] > vol_percentile_70, 2, 
                                 np.where(data['intraday_vol'] < vol_percentile_30, 0, 1))
    
    # Liquidity Momentum Core
    data['mid_price'] = (data['high'] + data['low']) / 2
    data['effective_spread'] = 2 * abs(data['close'] - data['mid_price']) / data['mid_price']
    data['volume_to_spread'] = data['volume'] / (data['effective_spread'] + 1e-8)
    
    data['liq_momentum_3'] = (data['volume_to_spread'] / data['volume_to_spread'].shift(3)) - 1
    data['liq_momentum_8'] = (data['volume_to_spread'] / data['volume_to_spread'].shift(8)) - 1
    
    # Price Efficiency & Microstructure Resilience
    # Market Microstructure Resilience
    up_days = data['close'] > data['open']
    down_days = data['close'] < data['open']
    
    data['price_recovery'] = np.where(up_days, 
                                     (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8), 
                                     0)
    data['downside_absorption'] = np.where(down_days, 
                                          (data['close'] - data['low']) / (abs(data['close'] - data['open']) + 1e-8), 
                                          0)
    data['resilience_score'] = (data['price_recovery'] + data['downside_absorption']) / 2
    
    # Regime-Adaptive Signal Construction
    # Volatility-Weighted Liquidity Momentum
    data['vol_weighted_liq'] = (data['liq_momentum_3'] - data['liq_momentum_8']) * (1 - data['effective_spread'])
    
    # Resilience-adjusted momentum
    data['resilience_adj_momentum'] = data['vol_weighted_liq'] * data['resilience_score']
    
    # Dynamic Confirmation & Enhancement
    # Spread-Momentum Divergence
    data['avg_spread_5'] = data['effective_spread'].rolling(window=5).mean()
    data['avg_spread_10'] = data['effective_spread'].rolling(window=10).mean()
    data['liq_spread_divergence'] = (data['liq_momentum_3'] - data['liq_momentum_8']) * ((data['avg_spread_5'] / data['avg_spread_10']) - 1)
    
    # Volume Acceleration Confirmation
    data['avg_volume_5'] = data['volume'].rolling(window=5).mean()
    data['volume_concentration'] = data['volume'] / (data['avg_volume_5'] + 1e-8)
    
    # Volume acceleration (2nd derivative approximation)
    data['volume_change_1'] = data['volume'] - data['volume'].shift(1)
    data['volume_change_2'] = data['volume_change_1'] - data['volume_change_1'].shift(1)
    data['volume_acceleration'] = data['volume_change_2']
    
    # Net concentration signal
    data['concentrated_buying'] = (data['volume_concentration'] > 1.2) & (data['close'] > data['open'])
    data['concentrated_selling'] = (data['volume_concentration'] > 1.2) & (data['close'] < data['open'])
    
    data['net_concentration'] = (data['concentrated_buying'].rolling(window=5).sum() - 
                                data['concentrated_selling'].rolling(window=5).sum()) / 5
    
    # Combined signal
    data['combined_signal'] = (data['resilience_adj_momentum'] + 
                              data['liq_spread_divergence'] + 
                              data['net_concentration'])
    
    # Final Alpha Generation
    # Stability-Weighted Signal
    data['price_stability'] = 1 - (abs(data['close'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8))
    data['signal_consistency'] = np.sign(data['combined_signal']) * np.sign(data['combined_signal'].shift(1))
    data['stability_weighted'] = (data['combined_signal'] * data['price_stability'] * 
                                 (1 + data['signal_consistency']))
    
    # Cross-Asset Regime Integration (simplified proxy)
    # Using volatility regime persistence as cross-asset confirmation proxy
    data['regime_persistence'] = (data['vol_regime'] == data['vol_regime'].shift(1)).rolling(window=5).mean()
    data['cross_asset_confirmation'] = data['regime_persistence'] * (1 - data['effective_spread'])
    
    # Volatility-normalized alpha
    data['vol_normalized_alpha'] = data['stability_weighted'] / (data['intraday_vol'] + 1e-8)
    
    # Final factor
    data['final_factor'] = data['vol_normalized_alpha'] * data['cross_asset_confirmation']
    
    # Clean up and return
    factor_series = data['final_factor'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return factor_series
