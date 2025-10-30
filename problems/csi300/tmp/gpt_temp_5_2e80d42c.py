import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Volatility-Efficiency Momentum with Volume-Price Alignment
    """
    data = df.copy()
    
    # Multi-Timeframe Efficiency Calculation
    # Short-Term Efficiency (3-day)
    data['close_ret_1d'] = data['close'].diff()
    data['short_efficiency'] = (data['close'] - data['close'].shift(3)) / (
        data['close_ret_1d'].abs().rolling(window=3, min_periods=3).sum()
    )
    
    # Medium-Term Efficiency (8-day)
    data['medium_efficiency'] = (data['close'] - data['close'].shift(8)) / (
        data['close_ret_1d'].abs().rolling(window=8, min_periods=8).sum()
    )
    
    # Long-Term Efficiency (21-day)
    data['long_efficiency'] = (data['close'] - data['close'].shift(21)) / (
        data['close_ret_1d'].abs().rolling(window=21, min_periods=21).sum()
    )
    
    # Volatility-Adjusted Momentum
    # Short-Term Return (t-5 to t)
    data['short_return'] = data['close'] / data['close'].shift(5) - 1
    
    # Medium-Term Return (t-20 to t)
    data['medium_return'] = data['close'] / data['close'].shift(20) - 1
    
    # Volatility Adjustment - High-Low Range Volatility (t-20 to t-1)
    data['daily_range'] = data['high'] - data['low']
    data['range_volatility'] = data['daily_range'].rolling(window=20, min_periods=20).mean()
    
    # Scale Returns by Volatility
    data['vol_adj_short_return'] = data['short_return'] / data['range_volatility']
    data['vol_adj_medium_return'] = data['medium_return'] / data['range_volatility']
    
    # Volume-Price Divergence Analysis
    # Volume Trend Slope (t-5 to t-1)
    data['volume_trend'] = data['volume'].rolling(window=5, min_periods=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else np.nan
    )
    
    # Volume-Price Correlation
    data['price_change_sign'] = np.sign(data['close'] - data['close'].shift(1))
    data['volume_change_sign'] = np.sign(data['volume'] - data['volume'].shift(1))
    data['volume_price_corr'] = data['price_change_sign'] * data['volume_change_sign']
    
    # Volume-Weighted Efficiency
    data['volume_weighted_efficiency'] = data['volume'] * data['medium_efficiency']
    
    # Efficiency-Momentum Alignment
    # Multi-Timeframe Consistency
    data['short_medium_align'] = np.sign(data['short_efficiency']) * np.sign(data['medium_efficiency'])
    data['medium_long_align'] = np.sign(data['medium_efficiency']) * np.sign(data['long_efficiency'])
    data['cross_timeframe_consistency'] = data['short_medium_align'] * data['medium_long_align']
    
    # Volatility-Momentum Integration
    data['vol_momentum_short'] = data['vol_adj_short_return'] * data['short_efficiency']
    data['vol_momentum_medium'] = data['vol_adj_medium_return'] * data['medium_efficiency']
    data['momentum_efficiency_product'] = (data['vol_momentum_short'] + data['vol_momentum_medium']) / 2
    
    # Volume-Efficiency Convergence
    data['volume_trend_efficiency'] = data['volume_trend'] * data['medium_efficiency']
    data['volume_cross_consistency'] = data['volume_price_corr'] * data['cross_timeframe_consistency']
    data['volume_weighted_momentum'] = data['volume_weighted_efficiency'] * data['vol_adj_medium_return']
    
    # Range-Based Efficiency Confirmation
    # Short-Term Range Efficiency
    data['short_range'] = data['high'].rolling(window=3, min_periods=3).max() - data['low'].rolling(window=3, min_periods=3).min()
    data['short_range_efficiency'] = (data['close'] - data['close'].shift(3)) / data['short_range']
    
    # Medium-Term Range Efficiency
    data['medium_range'] = data['high'].rolling(window=8, min_periods=8).max() - data['low'].rolling(window=8, min_periods=8).min()
    data['medium_range_efficiency'] = (data['close'] - data['close'].shift(8)) / data['medium_range']
    
    # Range-Efficiency Alignment
    data['range_efficiency_align'] = data['medium_range_efficiency'] * data['medium_efficiency']
    
    # Volatility Regime Detection
    # Volatility Spectrum
    data['short_term_vol'] = data['close_ret_1d'].abs().rolling(window=3, min_periods=3).sum() / data['short_range']
    data['medium_term_vol'] = data['close_ret_1d'].abs().rolling(window=8, min_periods=8).sum() / data['medium_range']
    data['volatility_ratio'] = data['short_term_vol'] / data['medium_term_vol']
    
    # Regime Classification
    data['medium_vol_ma'] = data['medium_term_vol'].rolling(window=5, min_periods=5).mean()
    data['vol_regime'] = np.where(
        data['medium_term_vol'] > 1.2 * data['medium_vol_ma'], 'high',
        np.where(data['medium_term_vol'] < 0.8 * data['medium_vol_ma'], 'low', 'normal')
    )
    
    # Efficiency Persistence Filtering
    data['direction_consistency'] = np.sign(data['medium_efficiency']).rolling(window=3, min_periods=3).sum() / 3
    data['volume_confirmation'] = data['volume_price_corr'] * data['direction_consistency']
    data['range_support'] = data['medium_range_efficiency'] * data['direction_consistency']
    
    # Core Momentum-Efficiency Factor
    data['core_momentum_efficiency'] = (
        data['vol_adj_medium_return'] * data['cross_timeframe_consistency'] +
        data['volume_weighted_efficiency'] * data['momentum_efficiency_product'] +
        data['range_efficiency_align'] * data['volume_confirmation']
    ) / 3
    
    # Regime-Adaptive Combination
    def apply_regime_weights(row):
        if row['vol_regime'] == 'high':
            short_weight = 0.7
            medium_weight = 0.2
            long_weight = 0.1
            volume_enhance = 1.5  # Amplify volume signals
        elif row['vol_regime'] == 'low':
            short_weight = 0.1
            medium_weight = 0.45
            long_weight = 0.45
            volume_enhance = 0.8  # Reduce volume noise
        else:  # normal
            short_weight = 0.33
            medium_weight = 0.33
            long_weight = 0.34
            volume_enhance = 1.0
        
        regime_factor = (
            short_weight * data.loc[row.name, 'vol_momentum_short'] +
            medium_weight * data.loc[row.name, 'vol_momentum_medium'] +
            long_weight * data.loc[row.name, 'long_efficiency']
        )
        
        volume_factor = data.loc[row.name, 'volume_cross_consistency'] * volume_enhance
        
        return regime_factor * volume_factor * data.loc[row.name, 'direction_consistency']
    
    data['regime_adaptive_factor'] = data.apply(apply_regime_weights, axis=1)
    
    # Multi-Dimensional Validation
    data['volume_price_alignment'] = data['volume_price_corr'] * data['volume_confirmation']
    data['range_momentum_support'] = data['range_efficiency_align'] * data['vol_adj_medium_return']
    data['cross_timeframe_confirmation'] = data['cross_timeframe_consistency'] * data['direction_consistency']
    
    # Final Alpha Construction
    data['final_alpha'] = (
        data['core_momentum_efficiency'] * 0.4 +
        data['regime_adaptive_factor'] * 0.3 +
        data['volume_price_alignment'] * 0.1 +
        data['range_momentum_support'] * 0.1 +
        data['cross_timeframe_confirmation'] * 0.1
    )
    
    return data['final_alpha']
