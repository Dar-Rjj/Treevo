import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Scale Efficiency Analysis
    # Intraday Range Efficiency
    data['intraday_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['efficiency_momentum_3d'] = data['intraday_efficiency'] / data['intraday_efficiency'].shift(3)
    data['efficiency_vol_5d'] = data['intraday_efficiency'].rolling(window=5).std()
    
    # Opening Gap Efficiency
    data['gap_magnitude'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_efficiency'] = (data['high'] - data['open']) / abs(data['open'] - data['close'].shift(1)).replace(0, np.nan)
    data['gap_extremeness'] = (data['gap_magnitude'] - data['gap_magnitude'].rolling(window=20).mean()) / data['gap_magnitude'].rolling(window=20).std()
    
    # Multi-Period Momentum Efficiency
    data['high_3d'] = data['high'].rolling(window=3).max()
    data['low_3d'] = data['low'].rolling(window=3).min()
    data['short_term_efficiency'] = (data['close'] - data['close'].shift(3)) / (data['high_3d'] - data['low_3d']).replace(0, np.nan)
    
    data['high_10d'] = data['high'].rolling(window=10).max()
    data['low_10d'] = data['low'].rolling(window=10).min()
    data['medium_term_efficiency'] = (data['close'] - data['close'].shift(10)) / (data['high_10d'] - data['low_10d']).replace(0, np.nan)
    
    data['high_20d'] = data['high'].rolling(window=20).max()
    data['low_20d'] = data['low'].rolling(window=20).min()
    data['long_term_efficiency'] = (data['close'] - data['close'].shift(20)) / (data['high_20d'] - data['low_20d']).replace(0, np.nan)
    
    # Volume-Pressure Dynamics
    # Volume Shock Analysis
    data['volume_shock_intensity'] = data['volume'] / data['volume'].shift(5)
    data['dollar_volume_pressure'] = data['amount'] / data['amount'].shift(5)
    
    volume_persistence = []
    for i in range(len(data)):
        if i >= 5:
            window = data['volume'].iloc[i-4:i+1]
            persistence = (window > window.shift(1)).sum()
            volume_persistence.append(persistence)
        else:
            volume_persistence.append(np.nan)
    data['volume_persistence'] = volume_persistence
    
    # Volume-Price Divergence
    data['price_momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['volume_trend_5d'] = data['volume'] / data['volume'].shift(5) - 1
    data['momentum_volume_divergence'] = np.sign(data['price_momentum_5d']) * np.sign(data['volume_trend_5d'])
    
    # Liquidity Context
    data['daily_liquidity_intensity'] = data['volume'] * data['amount']
    data['volume_efficiency'] = data['volume'] / (data['high'] - data['low']).replace(0, np.nan)
    data['liquidity_persistence'] = data['daily_liquidity_intensity'].rolling(window=5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
    
    # Dynamic Regime Classification
    # Volatility-Pressure Matrix
    data['returns'] = data['close'].pct_change()
    data['volatility_15d'] = data['returns'].rolling(window=15).std()
    data['volatility_momentum'] = data['volatility_15d'] / data['volatility_15d'].shift(5)
    data['volume_pressure_regime'] = data['volume_shock_intensity'] * data['dollar_volume_pressure']
    
    # Efficiency-Elasticity Analysis
    data['momentum_elasticity_divergence'] = data['short_term_efficiency'] * data['medium_term_efficiency'] * data['long_term_efficiency']
    
    # Calculate daily efficiency-volume correlation over 8 days
    efficiency_volume_corr = []
    for i in range(len(data)):
        if i >= 8:
            window_efficiency = data['intraday_efficiency'].iloc[i-7:i+1]
            window_volume = data['volume'].iloc[i-7:i+1]
            if len(window_efficiency) == len(window_volume) and len(window_efficiency) >= 2:
                corr = np.corrcoef(window_efficiency, window_volume)[0,1]
                efficiency_volume_corr.append(corr)
            else:
                efficiency_volume_corr.append(np.nan)
        else:
            efficiency_volume_corr.append(np.nan)
    data['efficiency_volume_corr_8d'] = efficiency_volume_corr
    
    # Regime Transition Detection
    data['pressure_accumulation'] = data['volume_shock_intensity'].rolling(window=3).sum()
    data['regime_shift_signal'] = data['pressure_accumulation'] * data['momentum_elasticity_divergence']
    
    # Regime-Adaptive Signal Processing
    # Efficiency-Momentum Integration
    data['multi_scale_efficiency_momentum'] = (data['short_term_efficiency'] + data['medium_term_efficiency'] + data['long_term_efficiency']) / 3
    data['volume_weighted_efficiency'] = data['intraday_efficiency'] * data['volume_shock_intensity']
    data['gap_driven_momentum'] = data['gap_efficiency'] * data['gap_magnitude']
    
    # Gap and Range Interaction Analysis
    data['intraday_up_momentum'] = data['high'] - data['open']
    data['intraday_down_momentum'] = data['open'] - data['low']
    data['range_utilization'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    data['gap_range_interaction'] = abs(data['gap_magnitude']) * data['intraday_efficiency']
    data['closing_pressure'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Signal Enhancement and Validation
    # Persistence Analysis
    efficiency_signal_persistence = []
    for i in range(len(data)):
        if i >= 5:
            window = data['intraday_efficiency'].iloc[i-4:i+1]
            persistence = (window > 0).sum()
            efficiency_signal_persistence.append(persistence)
        else:
            efficiency_signal_persistence.append(np.nan)
    data['efficiency_signal_persistence'] = efficiency_signal_persistence
    
    # Volatility Context
    data['recent_volatility'] = data['returns'].rolling(window=5).std()
    
    # Composite Alpha Generation
    # Core Signal Integration
    data['efficiency_momentum_score'] = (
        data['multi_scale_efficiency_momentum'] * 
        data['momentum_elasticity_divergence'] * 
        data['volume_weighted_efficiency']
    )
    
    # Dynamic Adjustment Factors
    data['volume_10d_avg'] = data['volume'].rolling(window=10).mean()
    data['volume_confirmation_weight'] = data['volume'] / data['volume_10d_avg']
    
    data['relative_price_position'] = (data['close'] - data['low_20d']) / (data['high_20d'] - data['low_20d']).replace(0, np.nan)
    
    # Contrarian logic for extreme positions
    data['contrarian_adjustment'] = 1 - 2 * abs(data['relative_price_position'] - 0.5)
    
    # Final Signal Processing
    # Combine all components with regime-adaptive weighting
    regime_weight = 1 / (1 + abs(data['regime_shift_signal']))
    
    # Composite alpha calculation
    composite_alpha = (
        data['efficiency_momentum_score'] * 0.3 +
        data['gap_driven_momentum'] * 0.2 +
        data['range_utilization'] * 0.15 +
        data['closing_pressure'] * 0.15 +
        data['momentum_volume_divergence'] * 0.1 +
        data['efficiency_signal_persistence'] * 0.05 +
        data['contrarian_adjustment'] * 0.05
    ) * regime_weight * data['volume_confirmation_weight']
    
    # Volatility adjustment
    volatility_adjustment = 1 / (1 + data['recent_volatility'])
    composite_alpha = composite_alpha * volatility_adjustment
    
    # Return the final alpha series
    return composite_alpha
