import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Efficiency Analysis
    # Intraday Efficiency Components
    data['range_efficiency'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['position_efficiency'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    data['reversal_efficiency'] = (data['open'] - data['close']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Multi-Day Efficiency Assessment
    data['close_5d_ago'] = data['close'].shift(5)
    data['high_5d_max'] = data['high'].rolling(window=5, min_periods=1).max()
    data['low_5d_min'] = data['low'].rolling(window=5, min_periods=1).min()
    data['range_efficiency_5d'] = np.abs(data['close'] - data['close_5d_ago']) / (data['high_5d_max'] - data['low_5d_min']).replace(0, np.nan)
    
    data['efficiency_consistency'] = data['range_efficiency'] / data['range_efficiency_5d'].replace(0, np.nan)
    data['efficiency_momentum'] = data['range_efficiency'] - data['range_efficiency'].shift(3)
    
    # Volatility-Contextual Efficiency
    data['hl_range'] = data['high'] - data['low']
    data['volatility_adj_efficiency'] = data['range_efficiency'] / data['hl_range'].rolling(window=10, min_periods=1).std()
    data['multi_timeframe_volatility'] = data['hl_range'].rolling(window=5, min_periods=1).std() / data['hl_range'].rolling(window=20, min_periods=1).std()
    
    # Volume-Amount Divergence System
    # Volume Momentum Analysis
    data['volume_median_5d'] = data['volume'].rolling(window=5, min_periods=1).median()
    data['volume_divergence'] = data['volume'] / data['volume_median_5d'].replace(0, np.nan)
    data['volume_strength'] = (data['volume'] - data['volume'].shift(5)) / data['volume'].shift(5).replace(0, np.nan) * 100
    
    # Amount-Based Trading Intensity
    data['implied_price'] = data['amount'] / data['volume'].replace(0, np.nan)
    data['trading_intensity'] = np.abs(data['implied_price'] - data['close']) / data['close'].replace(0, np.nan)
    data['amount_price_divergence'] = np.sign(data['implied_price'] - data['close']) * data['trading_intensity']
    data['breakout_strength'] = (data['close'] - (data['high'] + data['low']) / 2) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Volume-Amount Coherence
    data['volume_amount_aligned'] = (np.sign(data['volume_divergence'] - 1) == np.sign(data['amount_price_divergence'])).astype(float)
    data['volume_amount_divergent'] = (np.sign(data['volume_divergence'] - 1) != np.sign(data['amount_price_divergence'])).astype(float)
    data['confirmation_strength'] = (data['volume_divergence'] - 1) * data['amount_price_divergence']
    
    # Momentum Acceleration Framework
    # Price Momentum Alignment
    data['momentum_3d'] = data['close'] - data['close'].shift(3)
    data['momentum_7d'] = data['close'] - data['close'].shift(7)
    data['momentum_divergence'] = (np.sign(data['momentum_3d']) != np.sign(data['momentum_7d'])).astype(float)
    data['momentum_acceleration'] = (data['close'] - data['close'].shift(3)) - (data['close'].shift(1) - data['close'].shift(4))
    
    # Range Position Momentum
    data['position_change'] = data['position_efficiency'] - data['position_efficiency'].shift(1)
    data['position_acceleration'] = data['position_change'] - data['position_change'].shift(1)
    data['position_consistency_5d'] = data['position_efficiency'] - data['position_efficiency'].rolling(window=5, min_periods=1).mean()
    
    # Efficiency-Momentum Synthesis
    data['efficiency_driven_momentum'] = data['efficiency_momentum'] * data['momentum_3d']
    
    # Gap Context and Reversal Analysis
    data['gap_magnitude'] = np.abs(data['open'] - data['close'].shift(1))
    data['gap_relative_size'] = data['gap_magnitude'] / (data['high'].shift(1) - data['low'].shift(1)).replace(0, np.nan)
    data['gap_direction'] = np.sign(data['open'] - data['close'].shift(1))
    data['gap_reversal_strength'] = np.sign(data['close'] - data['open']) * data['gap_direction']
    data['gap_efficiency'] = data['range_efficiency'] * np.abs(data['gap_reversal_strength'])
    data['gap_volume_confirmation'] = (data['volume_divergence'] - 1) * data['gap_reversal_strength']
    data['gap_fill_momentum'] = (data['close'] - data['open']) * data['gap_direction']
    data['gap_efficiency_interaction'] = data['gap_reversal_strength'] * data['efficiency_momentum']
    data['gap_momentum_integration'] = data['gap_fill_momentum'] * data['position_change']
    
    # Efficiency-Price Divergence Framework
    data['price_change'] = data['close'] - data['close'].shift(1)
    data['price_change_std_10d'] = data['price_change'].rolling(window=10, min_periods=1).std()
    data['efficiency_price_divergence'] = np.abs(data['range_efficiency'] - data['price_change'] / data['price_change_std_10d'].replace(0, np.nan))
    data['volume_divergence_signal'] = (data['volume_divergence'] - 1) - np.abs(data['price_change']) / data['price_change_std_10d'].replace(0, np.nan)
    data['implied_price_mean_10d'] = data['implied_price'].rolling(window=10, min_periods=1).mean()
    data['amount_divergence'] = (data['implied_price'] / data['implied_price_mean_10d'].replace(0, np.nan)) - data['range_efficiency']
    
    # Range Rejection and Pressure Integration
    data['upper_rejection'] = np.where(data['close'] < data['open'], 
                                     (data['high'] - data['close']) / (data['high'] - data['low']).replace(0, np.nan), 0)
    data['lower_rejection'] = np.where(data['close'] > data['open'], 
                                     (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan), 0)
    data['current_rejection'] = data['upper_rejection'] + data['lower_rejection']
    data['current_rejection_ratio'] = data['current_rejection'] / data['current_rejection'].rolling(window=3, min_periods=1).max().replace(0, np.nan)
    
    # Intraday Pressure Components
    data['morning_pressure'] = data['high'] - data['open']
    data['afternoon_pressure'] = data['close'] - data['low']
    data['intraday_reversal'] = data['morning_pressure'] - data['afternoon_pressure']
    data['daily_pressure_magnitude'] = (data['close'] - data['open']) * data['volume']
    data['cumulative_pressure_5d'] = data['daily_pressure_magnitude'].rolling(window=5, min_periods=1).sum()
    data['pressure_momentum_alignment'] = np.sign(data['cumulative_pressure_5d']) * data['daily_pressure_magnitude']
    
    # Dynamic Regime Switching Framework
    data['volatility_regime'] = data['hl_range'].rolling(window=5, min_periods=1).std() / data['hl_range'].rolling(window=5, min_periods=1).std().rolling(window=20, min_periods=1).mean()
    data['efficiency_regime'] = data['range_efficiency'].rolling(window=5, min_periods=1).mean() / data['range_efficiency'].rolling(window=20, min_periods=1).mean()
    data['volume_regime'] = data['volume'].rolling(window=5, min_periods=1).mean() / data['volume'].rolling(window=20, min_periods=1).mean()
    
    # Regime-Adaptive Factor Weighting
    data['volatility_weighted_efficiency'] = data['range_efficiency'] / data['volatility_regime'].replace(0, np.nan)
    data['volume_weighted_momentum'] = data['momentum_3d'] * data['volume_regime']
    data['regime_specific_gap_response'] = data['gap_reversal_strength'] * data['efficiency_regime']
    data['cross_regime_alignment'] = data['volatility_regime'] * data['efficiency_regime'] * data['volume_regime']
    
    # Calculate volume persistence (correlation)
    def rolling_corr_volume(x):
        if len(x) < 2:
            return np.nan
        return pd.Series(x).corr(pd.Series(x).shift(1))
    
    data['volume_persistence'] = data['volume'].rolling(window=5, min_periods=2).apply(rolling_corr_volume, raw=True)
    
    # Composite Alpha Generation
    # Core Efficiency-Momentum Factor
    core_efficiency_momentum = (data['range_efficiency'] * 
                               (data['volume_divergence'] - 1) * 
                               data['momentum_3d'] * 
                               data['momentum_acceleration'] *
                               data['volume_persistence'].fillna(0) *
                               data['efficiency_consistency'].fillna(0))
    
    # Amount-Sentiment Reinforcement Factor
    amount_sentiment_reinforcement = (data['breakout_strength'] *
                                     data['amount_price_divergence'] *
                                     data['confirmation_strength'] *
                                     data['multi_timeframe_volatility'] *
                                     data['gap_reversal_strength'])
    
    # Efficiency-Price Divergence Factor
    efficiency_price_divergence_factor = (data['efficiency_price_divergence'] *
                                         data['volume_divergence_signal'] *
                                         (data['efficiency_price_divergence'] - data['efficiency_price_divergence'].shift(1)) /
                                         data['volatility_regime'].replace(0, np.nan))
    
    # Range Rejection-Pressure Factor
    range_rejection_pressure = (data['current_rejection_ratio'] *
                               data['cumulative_pressure_5d'] *
                               data['intraday_reversal'] *
                               data['pressure_momentum_alignment'])
    
    # Dynamic Regime-Adaptive Factor
    dynamic_regime_adaptive = (data['volatility_weighted_efficiency'] *
                              data['volume_weighted_momentum'] *
                              data['regime_specific_gap_response'] *
                              data['cross_regime_alignment'] *
                              (data['volatility_regime'] - data['volatility_regime'].shift(1)))
    
    # Final Alpha Synthesis
    base_composite = core_efficiency_momentum * amount_sentiment_reinforcement
    divergence_adjusted = base_composite * (1 + efficiency_price_divergence_factor.fillna(0))
    pressure_context = divergence_adjusted * (1 + range_rejection_pressure.fillna(0))
    regime_adapted = pressure_context * (1 + dynamic_regime_adaptive.fillna(0))
    
    # Final factor with multi-dimensional consistency check
    final_alpha = regime_adapted
    
    # Ensure no future data is used and return only the factor values
    return final_alpha
