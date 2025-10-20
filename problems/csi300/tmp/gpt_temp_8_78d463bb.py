import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Period Momentum Analysis
    data['ret_5d'] = data['close'].pct_change(5)
    data['ret_20d'] = data['close'].pct_change(20)
    data['ret_60d'] = data['close'].pct_change(60)
    
    # Momentum divergence patterns
    data['mom_accel_sm'] = data['ret_5d'] - data['ret_20d']
    data['mom_accel_ml'] = data['ret_20d'] - data['ret_60d']
    
    # Momentum sign alignment
    data['mom_pos_count'] = ((data['ret_5d'] > 0).astype(int) + 
                            (data['ret_20d'] > 0).astype(int) + 
                            (data['ret_60d'] > 0).astype(int))
    
    # Volume-Price Relationship Analysis
    data['price_range'] = (data['high'] - data['low']) / data['close']
    data['volume_scaled_range'] = data['price_range'] * data['volume']
    
    # Volume momentum and persistence
    data['vol_ret_5d'] = data['volume'].pct_change(5)
    
    # Volume autocorrelation (5-day)
    data['vol_lag1'] = data['volume'].shift(1)
    data['vol_lag2'] = data['volume'].shift(2)
    data['vol_lag3'] = data['volume'].shift(3)
    data['vol_lag4'] = data['volume'].shift(4)
    data['vol_autocorr'] = data[['volume', 'vol_lag1', 'vol_lag2', 'vol_lag3', 'vol_lag4']].corrwith(
        data['volume'], axis=1).fillna(0)
    
    # Volume streaks
    data['vol_change'] = data['volume'] > data['volume'].shift(1)
    data['vol_streak'] = data['vol_change'].groupby(
        (data['vol_change'] != data['vol_change'].shift(1)).cumsum()
    ).cumcount() + 1
    data['vol_streak'] = data['vol_streak'] * np.where(data['vol_change'], 1, -1)
    
    # Volume-price divergence
    data['mom_vol_divergence'] = np.sign(data['ret_5d']) != np.sign(data['vol_ret_5d'])
    data['volume_confirmation'] = (np.sign(data['ret_5d']) == np.sign(data['vol_ret_5d'])).astype(int)
    data['range_vol_consistency'] = data['price_range'].rolling(5).corr(data['volume']).fillna(0)
    
    # Volatility Regime Identification
    data['vol_5d'] = data['close'].pct_change().rolling(5).std()
    data['vol_20d'] = data['close'].pct_change().rolling(20).std()
    data['vol_ratio'] = data['vol_5d'] / data['vol_20d']
    
    # Regime classification
    vol_median_60d = data['vol_20d'].rolling(60).median()
    data['high_vol_regime'] = (data['vol_20d'] > vol_median_60d).astype(int)
    data['low_vol_regime'] = (data['vol_20d'] < vol_median_60d).astype(int)
    data['transition_regime'] = (data['vol_ratio'] > 1.5).astype(int)
    
    # Amount-Based Order Flow Analysis
    data['amt_ret_5d'] = data['amount'].pct_change(5)
    data['amt_vol_ratio'] = data['amount'] / data['volume']
    data['amt_vol_ratio_trend'] = data['amt_vol_ratio'].pct_change(5)
    
    # Large order detection (amount clustering)
    data['amt_zscore'] = (data['amount'] - data['amount'].rolling(20).mean()) / data['amount'].rolling(20).std()
    data['large_orders'] = (data['amt_zscore'] > 2).astype(int)
    
    # Order flow synchronization
    data['vol_amt_corr'] = data['volume'].rolling(5).corr(data['amount']).fillna(0)
    data['order_flow_divergence'] = np.sign(data['vol_ret_5d']) != np.sign(data['amt_ret_5d'])
    
    # Regime-Adaptive Signal Processing
    # Base momentum divergence component
    data['base_mom_div'] = (data['mom_accel_sm'] * 0.4 + 
                           data['mom_accel_ml'] * 0.3 + 
                           data['mom_pos_count'] * 0.3)
    
    # Volume confirmation scoring
    data['volume_score'] = (data['volume_confirmation'] * 0.4 + 
                           (1 - data['mom_vol_divergence'].astype(int)) * 0.3 + 
                           data['range_vol_consistency'] * 0.3)
    
    # Volatility regime adjustment
    # High volatility regime signals
    high_vol_signals = (data['base_mom_div'] * 0.6 + 
                       data['vol_streak'].abs() * 0.2 + 
                       data['volume_scaled_range'] * 0.2)
    
    # Low volatility regime signals
    low_vol_signals = (data['mom_accel_sm'] * 0.5 + 
                      data['mom_vol_divergence'].astype(int) * 0.3 + 
                      data['large_orders'] * 0.2)
    
    # Transition regime signals
    trans_signals = (data['mom_accel_sm'] * 0.4 + 
                    data['volume_scaled_range'] * 0.3 + 
                    (1 - data['order_flow_divergence'].astype(int)) * 0.3)
    
    # Apply regime-specific weighting
    data['regime_adj_mom'] = (data['high_vol_regime'] * high_vol_signals + 
                             data['low_vol_regime'] * low_vol_signals + 
                             data['transition_regime'] * trans_signals)
    
    # Order flow filtering
    data['order_flow_score'] = ((1 - data['order_flow_divergence'].astype(int)) * 0.5 + 
                               data['vol_amt_corr'] * 0.3 + 
                               data['large_orders'] * 0.2)
    
    # Final alpha synthesis
    data['alpha'] = (data['regime_adj_mom'] * 0.5 + 
                    data['volume_score'] * 0.3 + 
                    data['order_flow_score'] * 0.2)
    
    # Volatility scaling
    data['alpha_scaled'] = data['alpha'] / (data['vol_20d'] + 1e-8)
    
    # Multi-timeframe alignment scoring
    mom_alignment = ((data['ret_5d'] > 0) & (data['ret_20d'] > 0) & (data['ret_60d'] > 0)).astype(int) - \
                   ((data['ret_5d'] < 0) & (data['ret_20d'] < 0) & (data['ret_60d'] < 0)).astype(int)
    
    # Divergence intensity measurement
    divergence_intensity = (data['mom_accel_sm'].abs() + 
                           data['mom_accel_ml'].abs() + 
                           data['mom_vol_divergence'].astype(int))
    
    # Final composite alpha
    final_alpha = (data['alpha_scaled'] * 0.6 + 
                  mom_alignment * 0.2 + 
                  divergence_intensity * 0.2)
    
    # Clean up intermediate columns
    result = final_alpha.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return result
