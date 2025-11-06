import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Volatility Regime Identification
    # Daily returns
    data['daily_ret'] = data['close'] / data['close'].shift(1) - 1
    
    # Short-term volatility (5-day)
    data['short_term_vol'] = data['daily_ret'].rolling(window=5).std()
    
    # Medium-term volatility (20-day)
    data['medium_term_vol'] = data['daily_ret'].rolling(window=20).std()
    
    # Volatility regime classification
    data['vol_ratio'] = data['short_term_vol'] / data['medium_term_vol']
    data['vol_regime'] = 'normal'
    data.loc[data['vol_ratio'] > 1.2, 'vol_regime'] = 'high'
    data.loc[data['vol_ratio'] < 0.8, 'vol_regime'] = 'low'
    
    # Regime-Adaptive Efficiency Components
    # Short-term efficiency (5-day)
    data['high_5d'] = data['high'].rolling(window=5).max()
    data['low_5d'] = data['low'].rolling(window=5).min()
    data['true_range_5d'] = np.maximum(
        data['high_5d'] - data['low_5d'],
        np.maximum(
            np.abs(data['high_5d'] - data['close'].shift(5)),
            np.abs(data['low_5d'] - data['close'].shift(5))
        )
    )
    data['price_move_5d'] = np.abs(data['close'] - data['close'].shift(5))
    data['short_term_efficiency'] = data['price_move_5d'] / data['true_range_5d']
    
    # Medium-term efficiency (20-day)
    data['high_20d'] = data['high'].rolling(window=20).max()
    data['low_20d'] = data['low'].rolling(window=20).min()
    data['true_range_20d'] = np.maximum(
        data['high_20d'] - data['low_20d'],
        np.maximum(
            np.abs(data['high_20d'] - data['close'].shift(20)),
            np.abs(data['low_20d'] - data['close'].shift(20))
        )
    )
    data['price_move_20d'] = np.abs(data['close'] - data['close'].shift(20))
    data['medium_term_efficiency'] = data['price_move_20d'] / data['true_range_20d']
    
    # Efficiency acceleration
    data['efficiency_diff'] = data['medium_term_efficiency'] - data['short_term_efficiency']
    data['high_10d'] = data['high'].rolling(window=10).max()
    data['low_10d'] = data['low'].rolling(window=10).min()
    data['medium_term_range'] = data['high_10d'] - data['low_10d']
    data['short_term_range'] = data['high_5d'] - data['low_5d']
    data['efficiency_acceleration'] = data['efficiency_diff'] * (1 - data['short_term_range'] / data['medium_term_range'])
    
    # Regime-Specific Momentum Calculation
    # High volatility regime
    data['price_deviation'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    data['true_range_daily'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            np.abs(data['high'] - data['close'].shift(1)),
            np.abs(data['low'] - data['close'].shift(1))
        )
    )
    data['true_range_vol'] = data['true_range_daily'].rolling(window=10).std()
    data['high_vol_momentum'] = -1 * data['price_deviation'] / data['true_range_vol']
    
    # Low volatility regime
    data['trend_strength'] = (data['close'] - data['close'].shift(10)) / data['close'].shift(10)
    data['acceleration'] = (data['close'] / data['close'].shift(5)) - (data['close'].shift(5) / data['close'].shift(10))
    data['current_range'] = data['high'] - data['low']
    data['avg_range_5d'] = (data['high'] - data['low']).rolling(window=5).mean()
    data['low_vol_momentum'] = (data['trend_strength'] + data['acceleration']) * (data['current_range'] / data['avg_range_5d'])
    
    # Normal volatility regime
    data['short_term_momentum'] = (data['close'] / data['close'].shift(3)) - 1
    data['medium_term_momentum'] = (data['close'] / data['close'].shift(8)) - 1
    data['composite_momentum'] = 0.6 * data['short_term_momentum'] + 0.4 * data['medium_term_momentum']
    data['normal_vol_momentum'] = data['composite_momentum'] * data['short_term_efficiency'] * data['efficiency_acceleration']
    
    # Volume-Position Confirmation System
    # Volume momentum alignment
    data['avg_volume_5d'] = data['volume'].rolling(window=5).mean()
    data['volume_adjustment'] = data['volume'] / data['avg_volume_5d']
    
    # Volume-weighted 5-day return
    vol_weighted_returns = []
    for i in range(len(data)):
        if i >= 5:
            window_data = data.iloc[i-4:i+1]
            vol_weighted_return = (window_data['volume'] * (window_data['close'] / window_data['close'].shift(1) - 1)).sum() / window_data['volume'].sum()
            vol_weighted_returns.append(vol_weighted_return)
        else:
            vol_weighted_returns.append(np.nan)
    data['vol_weighted_return'] = vol_weighted_returns
    data['raw_5d_return'] = (data['close'] / data['close'].shift(5)) - 1
    data['volume_price_alignment'] = data['vol_weighted_return'] / data['raw_5d_return']
    
    # Multi-timeframe volume momentum
    data['ultra_short_vol_mom'] = data['volume'] / data['volume'].shift(2) - 1
    data['avg_volume_5d_early'] = data['volume'].rolling(window=5).mean()
    data['avg_volume_5d_late'] = data['volume'].shift(5).rolling(window=5).mean()
    data['short_term_vol_mom'] = data['avg_volume_5d_early'] / data['avg_volume_5d_late'] - 1
    
    # Position strength assessment
    data['intraday_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    data['resistance_distance'] = (data['high_20d'] - data['close']) / data['close']
    data['support_distance'] = (data['close'] - data['low_20d']) / data['close']
    data['position_volume_integration'] = data['intraday_position'] * data['volume_adjustment'] * (1 - data['resistance_distance'] + data['support_distance'])
    
    # Order flow confirmation
    data['signed_amount'] = data['amount'] * np.sign(data['close'] - data['close'].shift(1))
    data['cumulative_signed_amount'] = data['signed_amount'].rolling(window=5).sum()
    data['cumulative_abs_amount'] = data['amount'].rolling(window=5).sum()
    data['cumulative_imbalance'] = data['cumulative_signed_amount'] / data['cumulative_abs_amount']
    
    # Composite Alpha Generation
    # Primary factor construction
    regime_momentum = np.where(
        data['vol_regime'] == 'high', data['high_vol_momentum'],
        np.where(
            data['vol_regime'] == 'low', data['low_vol_momentum'],
            data['normal_vol_momentum']
        )
    )
    
    data['efficiency_momentum_core'] = data['efficiency_acceleration'] * regime_momentum
    data['volume_position_multiplier'] = data['volume_price_alignment'] * data['position_volume_integration']
    data['primary_factor'] = data['efficiency_momentum_core'] * data['volume_position_multiplier']
    
    # Order flow integration
    data['final_alpha'] = data['primary_factor'] * data['cumulative_imbalance']
    
    return data['final_alpha']
