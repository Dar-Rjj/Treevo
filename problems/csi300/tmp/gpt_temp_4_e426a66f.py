import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate returns for volatility
    data['returns'] = data['close'] / data['close'].shift(1) - 1
    
    # Multi-Timeframe Volatility Classification
    data['short_term_vol'] = data['returns'].rolling(window=20, min_periods=20).std()
    data['long_term_vol'] = data['returns'].rolling(window=60, min_periods=60).std()
    data['volatility_regime'] = ((data['high'].shift(1) - data['low'].shift(1)) / 
                                (data['high'].shift(5) - data['low'].shift(5))) * \
                               (data['short_term_vol'] / data['long_term_vol'])
    
    # Bidirectional Rejection-Flow Dynamics
    data['upper_rejection_strength'] = (data['high'] - np.maximum(data['open'], data['close'])) / (data['high'] - data['low'])
    data['lower_rejection_strength'] = (np.minimum(data['open'], data['close']) - data['low']) / (data['high'] - data['low'])
    data['net_rejection_pressure'] = data['upper_rejection_strength'] - data['lower_rejection_strength']
    data['volume_weighted_rejection'] = data['net_rejection_pressure'] * data['volume'] * \
                                       ((data['high'].shift(1) - data['low'].shift(1)) / 
                                        (data['high'].shift(5) - data['low'].shift(5)))
    
    # Volatility-Scaled Order Flow Velocity
    data['implicit_bid_pressure'] = ((data['close'] - data['low']) / (data['high'] - data['low'])) * data['volume'] * \
                                   ((data['high'].shift(1) - data['low'].shift(1)) / 
                                    (data['high'].shift(5) - data['low'].shift(5)))
    data['implicit_ask_pressure'] = ((data['high'] - data['close']) / (data['high'] - data['low'])) * data['volume'] * \
                                   ((data['high'].shift(1) - data['low'].shift(1)) / 
                                    (data['high'].shift(5) - data['low'].shift(5)))
    data['order_flow_imbalance'] = (data['implicit_bid_pressure'] - data['implicit_ask_pressure']) / data['volume']
    data['volatility_flow_velocity'] = data['order_flow_imbalance'] * \
                                      (np.abs(data['close'] - data['close'].shift(1)) / (data['high'] - data['low'])) * \
                                      ((data['high'].shift(1) - data['low'].shift(1)) / 
                                       (data['high'].shift(5) - data['low'].shift(5)))
    
    # Multi-Timeframe Efficiency-Velocity Divergence
    data['intraday_efficiency'] = (np.abs(data['close'] - data['open']) / (data['high'] - data['low'])) * \
                                 ((data['high'].shift(1) - data['low'].shift(1)) / 
                                  (data['high'].shift(5) - data['low'].shift(5)))
    data['true_range_efficiency'] = ((data['high'] - data['low']) / np.abs(data['close'].shift(1) - data['open'])) * \
                                   ((data['high'].shift(1) - data['low'].shift(1)) / 
                                    (data['high'].shift(5) - data['low'].shift(5)))
    data['efficiency_divergence'] = data['intraday_efficiency'] - data['true_range_efficiency']
    data['volatility_efficiency_momentum'] = ((data['intraday_efficiency'] / data['intraday_efficiency'].shift(1) - 1) * 
                                             ((data['high'].shift(1) - data['low'].shift(1)) / 
                                              (data['high'].shift(5) - data['low'].shift(5))))
    
    # Rejection-Flow Volatility Asymmetry
    data['upper_rejection_momentum'] = data['upper_rejection_strength'] * (data['close'] / data['close'].shift(1) - 1) * \
                                      ((data['high'].shift(1) - data['low'].shift(1)) / 
                                       (data['high'].shift(5) - data['low'].shift(5)))
    data['lower_rejection_momentum'] = data['lower_rejection_strength'] * (data['close'] / data['close'].shift(1) - 1) * \
                                      ((data['high'].shift(1) - data['low'].shift(1)) / 
                                       (data['high'].shift(5) - data['low'].shift(5)))
    data['rejection_momentum_spread'] = data['upper_rejection_momentum'] - data['lower_rejection_momentum']
    data['volume_confirmed_volatility_asymmetry'] = data['rejection_momentum_spread'] * data['volume'] * \
                                                   ((data['high'].shift(1) - data['low'].shift(1)) / 
                                                    (data['high'].shift(5) - data['low'].shift(5)))
    
    # Trade Size Velocity-Volatility Dynamics
    data['avg_trade_size'] = data['amount'] / data['volume']
    data['trade_size_momentum'] = (data['avg_trade_size'] / data['avg_trade_size'].shift(1)) - 1
    
    # Calculate institutional activity (5-day count of above-average trade size)
    avg_trade_size_5d = data['avg_trade_size'].rolling(window=5, min_periods=5).mean()
    institutional_count = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 4:
            count = sum(1 for j in range(i-4, i+1) if data['avg_trade_size'].iloc[j] > avg_trade_size_5d.iloc[j])
            institutional_count.iloc[i] = count / 5
        else:
            institutional_count.iloc[i] = np.nan
    data['institutional_activity'] = institutional_count
    
    data['volatility_trade_size_efficiency'] = data['avg_trade_size'] * data['intraday_efficiency'] * \
                                              ((data['high'].shift(1) - data['low'].shift(1)) / 
                                               (data['high'].shift(5) - data['low'].shift(5)))
    
    # Flow-Momentum Volatility Divergence Synthesis
    data['clean_momentum'] = data['close'] / data['close'].shift(1) - 1
    data['volatility_scaled_acceleration'] = ((data['close'] - data['close'].shift(3)) / data['close'].shift(3) - 
                                             (data['close'].shift(3) - data['close'].shift(6)) / data['close'].shift(6)) * \
                                            ((data['high'].shift(1) - data['low'].shift(1)) / 
                                             (data['high'].shift(5) - data['low'].shift(5)))
    data['momentum_efficiency_ratio'] = (data['clean_momentum'] / ((data['high'] - data['low']) / data['close'].shift(1))) * \
                                       ((data['high'].shift(1) - data['low'].shift(1)) / 
                                        (data['high'].shift(5) - data['low'].shift(5)))
    data['three_day_volatility_flow_momentum'] = ((data['close'] / data['close'].shift(3) - 1) * 
                                                 data['volume_weighted_rejection'] * 
                                                 ((data['high'].shift(1) - data['low'].shift(1)) / 
                                                  (data['high'].shift(5) - data['low'].shift(5))))
    
    # Volatility-Enhanced Breakout Dynamics
    data['opening_gap_volatility_pressure'] = (np.abs(data['open'] - data['close'].shift(1)) / (data['high'] - data['low'])) * \
                                             ((data['high'].shift(1) - data['low'].shift(1)) / 
                                              (data['high'].shift(5) - data['low'].shift(5)))
    
    # Volatility-Flow Divergence Patterns
    data['efficiency_volatility_flow_divergence'] = np.sign(data['volatility_efficiency_momentum']) * \
                                                   np.sign(data['volatility_flow_velocity'])
    data['rejection_absorption_volatility_alignment'] = np.sign(data['net_rejection_pressure']) * \
                                                       np.sign(data['trade_size_momentum']) * \
                                                       ((data['high'].shift(1) - data['low'].shift(1)) / 
                                                        (data['high'].shift(5) - data['low'].shift(5)))
    data['price_volatility_flow_divergence'] = np.sign(data['clean_momentum']) * np.sign(data['order_flow_imbalance'])
    data['trade_size_volatility_divergence'] = np.sign(data['trade_size_momentum']) * np.sign(data['volatility_flow_velocity'])
    
    # Range and Liquidity Volatility Assessment
    data['volatility_liquidity_absorption'] = (data['volume'] / (data['high'] - data['low'])) * \
                                             ((data['high'].shift(1) - data['low'].shift(1)) / 
                                              (data['high'].shift(5) - data['low'].shift(5)))
    data['absorption_volatility_momentum'] = data['volatility_liquidity_absorption'] / data['volatility_liquidity_absorption'].shift(1)
    
    # Multi-timeframe Volatility Persistence
    def calculate_persistence(series, window=3):
        persistence = pd.Series(index=series.index, dtype=float)
        for i in range(len(series)):
            if i >= window-1:
                count = 0
                for j in range(i-window+1, i+1):
                    if j > 0 and np.sign(series.iloc[j]) == np.sign(series.iloc[j-1]):
                        count += 1
                persistence.iloc[i] = count / window
            else:
                persistence.iloc[i] = np.nan
        return persistence
    
    data['volatility_flow_persistence'] = calculate_persistence(data['order_flow_imbalance'])
    data['volatility_efficiency_consistency'] = calculate_persistence(data['intraday_efficiency'].diff())
    data['rejection_volatility_persistence'] = calculate_persistence(data['net_rejection_pressure'])
    data['trade_size_volatility_persistence'] = calculate_persistence(data['trade_size_momentum'])
    
    # Core Volatility-Velocity Components
    data['volatility_flow_efficiency_velocity'] = data['volatility_flow_velocity'] * data['volatility_efficiency_momentum']
    data['rejection_volatility_flow_velocity'] = data['volume_confirmed_volatility_asymmetry'] * data['volatility_trade_size_efficiency']
    data['absorption_volatility_momentum_velocity'] = data['clean_momentum'] * data['absorption_volatility_momentum']
    data['breakout_volatility_flow_velocity'] = data['opening_gap_volatility_pressure'] * data['intraday_efficiency']
    
    # Microstructure-Volatility Confirmed Signals
    data['volatility_validated_flow'] = data['volatility_flow_efficiency_velocity'] * data['efficiency_volatility_flow_divergence']
    data['rejection_volatility_aligned_momentum'] = data['rejection_volatility_flow_velocity'] * data['rejection_absorption_volatility_alignment']
    data['institutional_volatility_flow_velocity'] = data['absorption_volatility_momentum_velocity'] * data['institutional_activity']
    data['range_enhanced_volatility_velocity'] = data['breakout_volatility_flow_velocity'] * \
                                                ((data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1)) > 1.2).astype(float) - \
                                                ((data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1)) < 0.8).astype(float)
    
    # Final Volatility-Flow Alpha Synthesis
    primary_factor = data['volatility_validated_flow'] * data['volatility_liquidity_absorption']
    secondary_factor = data['rejection_volatility_aligned_momentum'] * data['volatility_efficiency_consistency']
    tertiary_factor = data['institutional_volatility_flow_velocity'] * data['trade_size_volatility_divergence']
    quaternary_factor = data['range_enhanced_volatility_velocity'] * data['rejection_volatility_persistence']
    
    # Regime-weighted combination
    high_vol_weight = data['short_term_vol'] / data['long_term_vol']
    volume_surge_multiplier = data['volume'] / data['volume'].shift(5)
    
    # Composite alpha with regime adjustments
    composite_alpha = (
        primary_factor * high_vol_weight +
        secondary_factor * 1.0 +  # Normal regime weight
        tertiary_factor * volume_surge_multiplier +
        quaternary_factor * data['volatility_flow_persistence']
    )
    
    # Normalize and return
    alpha = composite_alpha
    alpha = (alpha - alpha.rolling(window=20, min_periods=20).mean()) / alpha.rolling(window=20, min_periods=20).std()
    
    return alpha
