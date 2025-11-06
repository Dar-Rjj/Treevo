import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Price-Volume Divergence with Regime Detection alpha factor
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate daily returns
    data['returns'] = data['close'].pct_change()
    
    # 1. Calculate Price-Volume Divergence
    # Price Trend Strength
    def calc_slope(series, window):
        """Calculate linear regression slope for given window"""
        slopes = np.full(len(series), np.nan)
        for i in range(window-1, len(series)):
            if i >= window-1:
                y = series.iloc[i-window+1:i+1].values
                x = np.arange(len(y))
                if len(y) == window and not np.any(np.isnan(y)):
                    slope, _, _, _, _ = stats.linregress(x, y)
                    slopes[i] = slope
        return slopes
    
    # Price slopes
    data['price_slope_10'] = calc_slope(data['close'], 10)
    data['price_slope_5'] = calc_slope(data['close'], 5)
    data['price_acceleration'] = np.abs(data['price_slope_5'] - data['price_slope_10'])
    data['price_trend_strength'] = np.abs(data['price_slope_10']) + data['price_acceleration']
    
    # Volume slopes
    data['volume_slope_10'] = calc_slope(data['volume'], 10)
    data['volume_slope_5'] = calc_slope(data['volume'], 5)
    data['volume_acceleration'] = np.abs(data['volume_slope_5'] - data['volume_slope_10'])
    data['volume_trend_strength'] = np.abs(data['volume_slope_10']) + data['volume_acceleration']
    
    # Divergence Score
    data['direction_alignment'] = np.sign(data['price_slope_10'] * data['volume_slope_10'])
    data['raw_divergence'] = data['price_trend_strength'] * data['volume_trend_strength'] * data['direction_alignment']
    
    # Normalize divergence by historical average
    data['divergence_ma'] = data['raw_divergence'].rolling(window=20, min_periods=10).mean()
    data['divergence_score'] = data['raw_divergence'] / (np.abs(data['divergence_ma']) + 1e-8)
    
    # 2. Detect Market Regime Changes
    # Volatility Regime
    data['volatility_20d'] = data['returns'].rolling(window=20, min_periods=10).std()
    data['volatility_60d'] = data['returns'].rolling(window=60, min_periods=30).std()
    data['vol_ratio'] = data['volatility_20d'] / (data['volatility_60d'] + 1e-8)
    data['high_vol_regime'] = (data['vol_ratio'] > 1.2).astype(int)
    
    # Trend Regime
    data['price_slope_20'] = calc_slope(data['close'], 20)
    data['positive_slopes'] = ((data['price_slope_5'] > 0).astype(int) + 
                              (data['price_slope_10'] > 0).astype(int) + 
                              (data['price_slope_20'] > 0).astype(int))
    slopes = data[['price_slope_5', 'price_slope_10', 'price_slope_20']].values
    data['slope_variance'] = np.nanvar(slopes, axis=1, ddof=1)
    data['strong_trend_regime'] = ((data['positive_slopes'] >= 2) | (data['positive_slopes'] <= 1)) & (data['slope_variance'] < 0.0001)
    
    # Regime Transition Detection
    data['vol_breakout'] = (data['volatility_20d'] > 2 * data['volatility_20d'].rolling(window=20, min_periods=10).mean()).astype(int)
    data['consecutive_breakouts'] = data['vol_breakout'].rolling(window=3).sum()
    
    # Slope sign changes
    data['slope_5_sign_change'] = (np.sign(data['price_slope_5']) != np.sign(data['price_slope_5'].shift(1))).astype(int)
    data['slope_10_sign_change'] = (np.sign(data['price_slope_10']) != np.sign(data['price_slope_10'].shift(1))).astype(int)
    data['reversal_confirmations'] = data['slope_5_sign_change'] + data['slope_10_sign_change']
    
    # Regime Stability Score
    data['volatility_stability'] = 1 / (1 + data['vol_ratio'].rolling(window=10).std())
    data['trend_stability'] = 1 / (1 + data['slope_variance'].rolling(window=10).mean())
    data['regime_stability'] = (data['volatility_stability'] + data['trend_stability']) / 2
    
    # 3. Analyze Order Flow Characteristics
    # Price-Volume Efficiency
    data['abs_return'] = np.abs(data['close'] / data['open'] - 1)
    data['amount_per_move'] = data['amount'] / (data['abs_return'] + 1e-8)
    
    # Volume concentration (simplified)
    data['volume_ma_5'] = data['volume'].rolling(window=5).mean()
    data['volume_ratio'] = data['volume'] / (data['volume_ma_5'] + 1e-8)
    data['high_volume_day'] = (data['volume_ratio'] > 1.5).astype(int)
    
    # Supply-Demand Imbalance
    data['price_range'] = (data['high'] - data['low']) / data['open']
    data['net_move'] = np.abs(data['close'] - data['open']) / data['open']
    data['range_efficiency'] = data['net_move'] / (data['price_range'] + 1e-8)
    
    # Absorption patterns
    data['rejection_volume'] = data['volume'] * (1 - data['range_efficiency'])
    data['absorption_strength'] = data['rejection_volume'].rolling(window=5).mean()
    
    # Market Depth Quality
    data['avg_trade_size'] = data['amount'] / (data['volume'] + 1e-8)
    data['trade_size_var'] = data['avg_trade_size'].rolling(window=10).std()
    
    # Order Flow Persistence
    data['volume_persistence'] = (data['volume_ratio'].rolling(window=3).std() < 0.3).astype(int)
    
    # Market Microstructure Score
    data['efficiency_score'] = 1 / (1 + np.log1p(data['amount_per_move']))
    data['absorption_score'] = 1 / (1 + data['absorption_strength'])
    data['depth_score'] = 1 / (1 + data['trade_size_var'])
    data['microstructure_score'] = (data['efficiency_score'] + data['absorption_score'] + data['depth_score']) / 3
    data['weighted_microstructure'] = data['microstructure_score'] * data['regime_stability']
    
    # 4. Generate Adaptive Alpha Signal
    # Regime-based weighting
    data['volatility_weight'] = np.where(data['high_vol_regime'] == 1, 0.7, 1.0)
    data['trend_weight'] = np.where(data['strong_trend_regime'], 1.2, 0.8)
    data['transition_penalty'] = np.where(data['reversal_confirmations'] > 0, 0.6, 1.0)
    
    # Combine components
    data['weighted_divergence'] = (data['divergence_score'] * 
                                  data['volatility_weight'] * 
                                  data['trend_weight'] * 
                                  data['transition_penalty'])
    
    # Incorporate order flow insights
    data['order_flow_enhanced'] = data['weighted_divergence'] * data['weighted_microstructure'] * data['volume_persistence']
    
    # Final alpha factor with non-linear transformation
    data['alpha_raw'] = data['order_flow_enhanced']
    data['alpha_factor'] = np.tanh(data['alpha_raw'] / (data['alpha_raw'].rolling(window=20).std() + 1e-8))
    
    # Handle NaN values
    alpha_factor = data['alpha_factor'].fillna(0)
    
    return alpha_factor
