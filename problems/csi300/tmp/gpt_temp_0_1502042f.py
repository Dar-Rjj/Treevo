import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volume-Price Fractal Efficiency factor combining multi-timeframe alignment analysis
    with bid-ask pressure imbalance and volatility-regime adaptive momentum.
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Volume-Price Fractal Efficiency Components
    # Short-term alignment (1-3 days)
    data['vol_price_momentum_3d'] = ((data['close'] - data['close'].shift(3)) / data['close'].shift(3)) * data['volume']
    data['cum_vol_weighted_return_3d'] = data['vol_price_momentum_3d'].rolling(window=3, min_periods=1).sum()
    
    # Directional consistency measure
    data['price_change_3d'] = data['close'] - data['close'].shift(3)
    data['vol_change_3d'] = data['volume'] - data['volume'].shift(3)
    data['direction_agreement_3d'] = np.sign(data['price_change_3d']) * np.sign(data['vol_change_3d'])
    data['alignment_strength_3d'] = data['direction_agreement_3d'] * (abs(data['price_change_3d']) / data['close'].shift(3))
    
    # Medium-term alignment (5-10 days)
    # 10-day correlation between price changes and volume (excluding current day)
    data['price_ret_1d'] = data['close'].pct_change()
    vol_price_corr = []
    for i in range(len(data)):
        if i >= 10:
            # Use past 10 days excluding current
            price_changes = data['price_ret_1d'].iloc[i-10:i].values
            volumes = data['volume'].iloc[i-10:i].values
            if len(price_changes) > 1 and len(volumes) > 1:
                corr = np.corrcoef(price_changes, volumes)[0, 1]
                vol_price_corr.append(corr if not np.isnan(corr) else 0)
            else:
                vol_price_corr.append(0)
        else:
            vol_price_corr.append(0)
    data['vol_price_corr_10d'] = vol_price_corr
    
    # Volume acceleration during trends
    data['vol_growth_5d'] = data['volume'].rolling(window=5, min_periods=1).mean() / data['volume'].rolling(window=10, min_periods=1).mean()
    data['price_trend_5d'] = data['close'].rolling(window=5, min_periods=1).mean() / data['close'].rolling(window=10, min_periods=1).mean() - 1
    
    # Long-term context (20+ days)
    data['vol_regime_20d'] = data['volume'].rolling(window=20, min_periods=1).std() / data['volume'].rolling(window=20, min_periods=1).mean()
    data['price_efficiency_20d'] = (data['close'] - data['close'].rolling(window=20, min_periods=1).min()) / (data['close'].rolling(window=20, min_periods=1).max() - data['close'].rolling(window=20, min_periods=1).min())
    
    # Detect Fractal Inefficiency Patterns
    data['new_high_divergence'] = ((data['close'] > data['close'].rolling(window=10, min_periods=1).max()) & 
                                  (data['volume'] < data['volume'].rolling(window=10, min_periods=1).mean())).astype(int)
    data['new_low_divergence'] = ((data['close'] < data['close'].rolling(window=10, min_periods=1).min()) & 
                                 (data['volume'] > data['volume'].rolling(window=10, min_periods=1).mean())).astype(int)
    
    # Multi-scale momentum conflicts
    data['momentum_short'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_medium'] = data['close'] / data['close'].shift(10) - 1
    data['momentum_conflict'] = np.sign(data['momentum_short']) != np.sign(data['momentum_medium'])
    
    # Bid-Ask Pressure Imbalance Components
    data['intraday_position'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    data['closing_pressure'] = data['intraday_position'] * data['volume']
    
    # Pressure accumulation
    data['upper_range_close'] = (data['intraday_position'] > 0.7).astype(int)
    data['lower_range_close'] = (data['intraday_position'] < 0.3).astype(int)
    data['buy_pressure_cluster'] = data['upper_range_close'].rolling(window=5, min_periods=1).sum()
    data['sell_pressure_cluster'] = data['lower_range_close'].rolling(window=5, min_periods=1).sum()
    data['pressure_imbalance'] = data['buy_pressure_cluster'] - data['sell_pressure_cluster']
    
    # Pressure gradient
    data['pressure_gradient'] = data['pressure_imbalance'].diff(3)
    
    # Volatility-Regime Adaptive Momentum
    # Multi-scale volatility
    data['volatility_5d'] = data['close'].pct_change().rolling(window=5, min_periods=1).std()
    data['volatility_15d'] = data['close'].pct_change().rolling(window=15, min_periods=1).std()
    data['volatility_30d'] = data['close'].pct_change().rolling(window=30, min_periods=1).std()
    
    # Volatility regime classification
    data['vol_regime'] = np.where(data['volatility_15d'] > data['volatility_30d'] * 1.5, 'high',
                                 np.where(data['volatility_15d'] < data['volatility_30d'] * 0.7, 'low', 'normal'))
    
    # Regime-adaptive momentum
    data['momentum_5d_vol_adj'] = (data['close'] / data['close'].shift(5) - 1) / (data['volatility_5d'] + 1e-8)
    data['momentum_15d_vol_adj'] = (data['close'] / data['close'].shift(15) - 1) / (data['volatility_15d'] + 1e-8)
    
    # Dynamic weighting based on volatility regime
    data['short_term_weight'] = np.where(data['vol_regime'] == 'high', 0.6,
                                       np.where(data['vol_regime'] == 'low', 0.3, 0.45))
    data['medium_term_weight'] = 1 - data['short_term_weight']
    
    # Combine all components into final factor
    # Volume-Price Fractal Efficiency Score
    vpfe_score = (0.3 * data['alignment_strength_3d'].fillna(0) + 
                  0.25 * data['vol_price_corr_10d'].fillna(0) + 
                  0.2 * (1 - data['new_high_divergence'].fillna(0) - data['new_low_divergence'].fillna(0)) + 
                  0.15 * (1 - data['momentum_conflict'].fillna(0).astype(float)) + 
                  0.1 * data['price_efficiency_20d'].fillna(0))
    
    # Pressure Imbalance Factor
    pressure_factor = (0.4 * data['pressure_imbalance'].fillna(0) + 
                      0.35 * data['pressure_gradient'].fillna(0) + 
                      0.25 * data['closing_pressure'].fillna(0))
    
    # Adaptive Momentum Score
    momentum_score = (data['short_term_weight'] * data['momentum_5d_vol_adj'].fillna(0) + 
                     data['medium_term_weight'] * data['momentum_15d_vol_adj'].fillna(0))
    
    # Final combined factor
    fractal_efficiency_factor = (0.4 * vpfe_score + 
                                0.35 * pressure_factor + 
                                0.25 * momentum_score)
    
    return pd.Series(fractal_efficiency_factor, index=data.index)
