import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Price Efficiency with Volume Confirmation and Regime Switching
    """
    data = df.copy()
    
    # Calculate True Range
    data['prev_close'] = data['close'].shift(1)
    data['tr_high'] = data[['high', 'prev_close']].max(axis=1)
    data['tr_low'] = data[['low', 'prev_close']].min(axis=1)
    data['true_range'] = data['tr_high'] - data['tr_low']
    
    # Single-day Price Efficiency
    data['daily_efficiency'] = abs(data['close'] - data['prev_close']) / data['true_range']
    data['daily_efficiency'] = data['daily_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    # Multi-Timeframe Efficiency Ratios
    # 3-day efficiency
    data['close_3d_ago'] = data['close'].shift(3)
    data['tr_sum_3d'] = data['true_range'].rolling(window=3, min_periods=3).sum()
    data['efficiency_3d'] = abs(data['close'] - data['close_3d_ago']) / data['tr_sum_3d']
    
    # 5-day efficiency
    data['close_5d_ago'] = data['close'].shift(5)
    data['tr_sum_5d'] = data['true_range'].rolling(window=5, min_periods=5).sum()
    data['efficiency_5d'] = abs(data['close'] - data['close_5d_ago']) / data['tr_sum_5d']
    
    # Efficiency acceleration
    data['efficiency_accel'] = data['efficiency_3d'] / data['efficiency_5d']
    data['efficiency_accel'] = data['efficiency_accel'].replace([np.inf, -np.inf], np.nan)
    
    # Combined Efficiency Signal
    data['efficiency_signal'] = (data['daily_efficiency'] * 0.4 + 
                                data['efficiency_3d'] * 0.35 + 
                                data['efficiency_accel'] * 0.25)
    
    # Volume Confirmation System
    # Directional volume
    data['price_direction'] = np.sign(data['close'] - data['open'])
    data['directional_volume'] = data['volume'] * data['price_direction']
    
    # Cumulative directional volume over 3 days
    data['cum_directional_volume_3d'] = data['directional_volume'].rolling(window=3, min_periods=3).sum()
    
    # Volume-price alignment
    data['price_change_3d'] = data['close'] - data['close'].shift(3)
    data['volume_price_alignment'] = data['cum_directional_volume_3d'] * data['price_change_3d']
    
    # Volume regime detection
    data['volume_volatility'] = data['volume'].rolling(window=5, min_periods=5).std()
    data['volume_avg_5d'] = data['volume'].rolling(window=5, min_periods=5).mean()
    data['volume_level_ratio'] = data['volume'] / data['volume_avg_5d']
    
    # Volume regime strength
    data['volume_regime_strength'] = (data['volume_level_ratio'] * 
                                     data['volume_volatility'] / data['volume_avg_5d'])
    
    # Volume confirmation signal
    data['volume_confirmation'] = data['volume_price_alignment'] * data['volume_regime_strength']
    
    # Opening Auction Strength Analysis
    # Opening range (using first hour high-low as proxy - in practice would need intraday data)
    # For daily data, we'll use the daily range as opening range proxy
    data['opening_range'] = data['high'] - data['low']
    data['close_position_in_range'] = (data['close'] - data['low']) / data['opening_range']
    data['close_position_in_range'] = data['close_position_in_range'].replace([np.inf, -np.inf], np.nan)
    
    # Gap fill analysis
    data['gap_size'] = abs(data['open'] - data['prev_close'])
    data['gap_fill_percentage'] = abs(data['close'] - data['open']) / data['gap_size']
    data['gap_fill_percentage'] = data['gap_fill_percentage'].replace([np.inf, -np.inf], np.nan)
    
    # Auction failure detection
    data['auction_failure'] = np.where(data['gap_fill_percentage'] > 0.9, -1, 1)
    
    # Auction signal
    data['auction_signal'] = data['close_position_in_range'] * data['auction_failure']
    
    # Regime Switching Detection
    # Price-volume regime classification
    efficiency_threshold = data['efficiency_signal'].rolling(window=20, min_periods=20).median()
    volume_conf_threshold = data['volume_confirmation'].rolling(window=20, min_periods=20).median()
    
    # Regime classification
    data['trending_regime'] = ((data['efficiency_signal'] > efficiency_threshold) & 
                              (data['volume_confirmation'] > volume_conf_threshold)).astype(int)
    
    data['choppy_regime'] = ((data['efficiency_signal'] < efficiency_threshold) & 
                            (data['volume_confirmation'] < volume_conf_threshold)).astype(int)
    
    data['exhaustion_regime'] = ((data['efficiency_signal'] > efficiency_threshold) & 
                                (data['volume_confirmation'] < -volume_conf_threshold)).astype(int)
    
    # Timeframe regime consistency
    data['efficiency_1d_rank'] = data['daily_efficiency'].rolling(window=20, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    
    data['efficiency_3d_rank'] = data['efficiency_3d'].rolling(window=20, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    
    data['efficiency_5d_rank'] = data['efficiency_5d'].rolling(window=20, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    
    # Regime stability score
    data['regime_stability'] = (abs(data['efficiency_1d_rank'] - data['efficiency_3d_rank']) + 
                               abs(data['efficiency_3d_rank'] - data['efficiency_5d_rank'])) / 2
    
    # Regime signal
    data['regime_signal'] = (data['trending_regime'] * 1.0 + 
                            data['choppy_regime'] * -0.5 + 
                            data['exhaustion_regime'] * -1.0)
    
    # Adaptive Alpha Generation
    # Signal integration with regime-aware weighting
    regime_stability_weight = 1 - data['regime_stability']
    volume_confidence = abs(data['volume_confirmation']) / data['volume_confirmation'].abs().rolling(window=20, min_periods=20).mean()
    
    # Final alpha factor
    alpha = (data['efficiency_signal'] * regime_stability_weight * 0.4 +
            data['volume_confirmation'] * volume_confidence * 0.3 +
            data['auction_signal'] * 0.2 +
            data['regime_signal'] * 0.1)
    
    # Clean up and return
    alpha = alpha.replace([np.inf, -np.inf], np.nan)
    alpha = alpha.fillna(method='ffill').fillna(0)
    
    return alpha
