import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Divergence with Regime Detection alpha factor
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Price Trend Strength
    data['close_slope_10'] = data['close'].diff(5) / 5  # 10-day slope approximation
    data['close_slope_5'] = data['close'].diff(3) / 3   # 5-day slope approximation
    data['price_slope_diff'] = abs(data['close_slope_5'] - data['close_slope_10'])
    data['price_trend_strength'] = data['close_slope_10'].abs() * (1 + data['price_slope_diff'])
    
    # Volume Trend Strength
    data['volume_slope_10'] = data['volume'].diff(5) / 5  # 10-day volume slope
    data['volume_slope_5'] = data['volume'].diff(3) / 3   # 5-day volume slope
    data['volume_slope_diff'] = abs(data['volume_slope_5'] - data['volume_slope_10'])
    data['volume_trend_strength'] = data['volume_slope_10'].abs() * (1 + data['volume_slope_diff'])
    
    # Divergence Score
    data['divergence_raw'] = data['price_trend_strength'] * data['volume_trend_strength']
    data['direction_alignment'] = np.sign(data['close_slope_10'] * data['volume_slope_10'])
    data['divergence_score'] = data['divergence_raw'] * data['direction_alignment']
    
    # Market Regime Analysis
    # Volatility Regime
    data['volatility_20'] = data['close'].pct_change().rolling(window=20).std()
    data['volatility_60'] = data['close'].pct_change().rolling(window=60).std()
    data['vol_ratio'] = data['volatility_20'] / data['volatility_60']
    data['high_vol_regime'] = (data['vol_ratio'] > 1.2).astype(int)
    data['low_vol_regime'] = (data['vol_ratio'] < 0.8).astype(int)
    
    # Trend Regime
    data['trend_short'] = data['close'].rolling(window=5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
    data['trend_medium'] = data['close'].rolling(window=20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
    data['trend_consistency'] = np.sign(data['trend_short'] * data['trend_medium'])
    data['trend_strength'] = (abs(data['trend_short']) + abs(data['trend_medium'])) / 2
    
    # Regime Transition Detection
    data['vol_breakout'] = ((data['vol_ratio'] > 1.5) & (data['vol_ratio'].shift(1) <= 1.5)).astype(int)
    data['trend_reversal'] = ((data['trend_consistency'] < 0) & (data['trend_consistency'].shift(1) > 0)).astype(int)
    
    # Order Flow Analysis
    # Price-Volume Efficiency
    data['daily_range'] = data['high'] - data['low']
    data['price_move_efficiency'] = data['amount'] / (data['daily_range'].replace(0, np.nan) + 1e-6)
    data['volume_concentration'] = data['volume'] / data['volume'].rolling(window=5).mean()
    
    # Absorption Detection
    data['range_volume_ratio'] = data['daily_range'] / (data['volume'].replace(0, np.nan) + 1e-6)
    data['supply_demand_imbalance'] = (data['close'] - data['open']) / (data['daily_range'].replace(0, np.nan) + 1e-6)
    data['absorption_zone'] = ((data['volume'] > data['volume'].rolling(window=10).mean() * 1.5) & 
                              (abs(data['supply_demand_imbalance']) < 0.3)).astype(int)
    
    # Market Depth Assessment
    data['trade_size_indicator'] = data['amount'] / (data['volume'].replace(0, np.nan) + 1e-6)
    data['order_flow_persistence'] = data['trade_size_indicator'].rolling(window=5).std()
    
    # Adaptive Alpha Generation
    # Regime-based signal weighting
    data['regime_weight'] = np.where(data['high_vol_regime'] == 1, 0.7, 
                                   np.where(data['low_vol_regime'] == 1, 1.3, 1.0))
    
    # Trend regime adjustment
    data['trend_weight'] = np.where(data['trend_strength'] > data['trend_strength'].quantile(0.7), 1.2, 1.0)
    
    # Order flow integration
    data['order_flow_score'] = (data['price_move_efficiency'].rank(pct=True) - 
                               data['absorption_zone'] * 0.3 + 
                               data['order_flow_persistence'].rank(pct=True))
    
    # Final factor combination
    data['base_factor'] = data['divergence_score'] * data['regime_weight'] * data['trend_weight']
    data['enhanced_factor'] = data['base_factor'] + data['order_flow_score'] * 0.2
    
    # Handle transition periods
    data['transition_boost'] = np.where((data['vol_breakout'] == 1) | (data['trend_reversal'] == 1), 1.5, 1.0)
    data['final_factor'] = data['enhanced_factor'] * data['transition_boost']
    
    # Normalize and rank
    factor = data['final_factor'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return factor
