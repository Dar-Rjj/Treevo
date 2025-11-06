import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Hierarchical Range-Volume Efficiency Factor with Structural Regime Integration
    """
    data = df.copy()
    
    # Multi-Timeframe Range Efficiency Analysis
    # Daily Range Efficiency
    data['daily_range_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['daily_range_efficiency'] = data['daily_range_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    # Short-term Range Efficiency (5-day average)
    data['short_term_efficiency'] = data['daily_range_efficiency'].rolling(window=5, min_periods=1).mean()
    
    # Medium-term Range Efficiency (10-day average)
    data['medium_term_efficiency'] = data['daily_range_efficiency'].rolling(window=10, min_periods=1).mean()
    
    # Range Efficiency Persistence
    data['efficiency_persistence'] = data['daily_range_efficiency'].rolling(window=10, min_periods=1).apply(
        lambda x: (x > 0).sum(), raw=True
    )
    
    # Efficiency Divergence Patterns
    data['efficiency_divergence'] = data['short_term_efficiency'] - data['medium_term_efficiency']
    data['efficiency_momentum'] = data['short_term_efficiency'] - data['short_term_efficiency'].shift(5)
    
    # Volume-Weighted Directional Pressure
    # Bidirectional Flow Analysis
    data['buy_pressure'] = data['volume'] * (data['close'] - data['low']) / (data['high'] - data['low'])
    data['buy_pressure'] = data['buy_pressure'].replace([np.inf, -np.inf], np.nan)
    
    data['sell_pressure'] = data['volume'] * (data['high'] - data['close']) / (data['high'] - data['low'])
    data['sell_pressure'] = data['sell_pressure'].replace([np.inf, -np.inf], np.nan)
    
    data['net_pressure'] = data['buy_pressure'] - data['sell_pressure']
    data['net_pressure_accumulation'] = data['net_pressure'].rolling(window=5, min_periods=1).sum()
    
    # Volume Confirmation Patterns
    data['volume_median_5d'] = data['volume'].rolling(window=5, min_periods=1).median()
    data['volume_spike'] = (data['volume'] > data['volume_median_5d']).astype(int)
    
    data['volume_efficiency_alignment'] = np.where(
        data['daily_range_efficiency'] > 0, data['volume'], 0
    )
    
    def count_consecutive_volume(series):
        count = 0
        result = []
        for val in series:
            if val > series.rolling(window=5, min_periods=1).median().iloc[0] if len(series) >= 5 else val > series.mean():
                count += 1
            else:
                count = 0
            result.append(count)
        return result
    
    data['volume_persistence'] = count_consecutive_volume(data['volume'])
    
    # Microstructural Efficiency Scoring
    data['price_discovery_efficiency'] = (data['high'] - data['low']) / (data['amount'] / data['volume'])
    data['price_discovery_efficiency'] = data['price_discovery_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    data['volume_clustering_efficiency'] = (
        data['volume'].rolling(window=5, min_periods=1).std() / 
        data['volume'].rolling(window=5, min_periods=1).mean()
    )
    data['volume_clustering_efficiency'] = data['volume_clustering_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    def rolling_corr_5d(series1, series2):
        return series1.rolling(window=5, min_periods=1).corr(series2)
    
    data['information_absorption_rate'] = rolling_corr_5d(
        abs(data['close'] - data['close'].shift(1)), 
        data['volume']
    )
    
    # Structural Regime Detection
    # Volatility-Regime Classification
    data['close_returns_5d'] = data['close'].diff()
    data['close_returns_20d'] = data['close'].diff()
    
    data['regime_volatility_ratio'] = (
        data['close_returns_5d'].rolling(window=5, min_periods=1).std() / 
        data['close_returns_20d'].rolling(window=20, min_periods=1).std()
    )
    data['regime_volatility_ratio'] = data['regime_volatility_ratio'].replace([np.inf, -np.inf], np.nan)
    
    data['volume_volatility_alignment'] = (
        np.sign(data['volume'] - data['volume'].shift(1)) * 
        np.sign(
            data['close_returns_5d'].rolling(window=5, min_periods=1).std() - 
            data['close_returns_5d'].rolling(window=5, min_periods=1).std().shift(1)
        )
    )
    
    data['regime_persistence'] = rolling_corr_5d(
        data['daily_range_efficiency'].shift(2), 
        data['daily_range_efficiency'].shift(5)
    )
    
    # Liquidity Depth Analysis
    data['effective_spread_depth'] = (data['high'] - data['low']) * data['volume'] / data['amount']
    data['effective_spread_depth'] = data['effective_spread_depth'].replace([np.inf, -np.inf], np.nan)
    
    data['volume_concentration'] = (
        data['volume'].rolling(window=5, min_periods=1).max() / 
        data['volume'].rolling(window=5, min_periods=1).sum()
    )
    
    data['depth_resilience'] = 1 - (abs(data['close'] - data['close'].shift(1)) / data['effective_spread_depth'])
    data['depth_resilience'] = data['depth_resilience'].replace([np.inf, -np.inf], np.nan)
    
    # Structural Break Detection
    data['volume_avg_10d'] = data['volume'].rolling(window=10, min_periods=1).mean()
    data['volume_avg_50d'] = data['volume'].rolling(window=50, min_periods=1).mean()
    
    data['volume_break'] = (
        (data['volume'] > 2 * data['volume_avg_10d']) & 
        (data['volume'] > 1.5 * data['volume_avg_50d'])
    ).astype(int)
    
    data['price_break'] = (
        abs(data['close'] - data['close'].shift(1)) > 
        2 * data['close_returns_20d'].rolling(window=20, min_periods=1).std()
    ).astype(int)
    
    # Cross-Dimensional Integration
    # Efficiency-Pressure Coupling
    data['momentum_synchronization'] = rolling_corr_5d(
        data['daily_range_efficiency'], 
        data['net_pressure_accumulation']
    )
    
    data['lead_lag_relationship'] = (
        rolling_corr_5d(data['daily_range_efficiency'], data['volume'].shift(-1)) - 
        rolling_corr_5d(data['daily_range_efficiency'], data['volume'].shift(1))
    )
    
    data['coupling_strength'] = abs(data['momentum_synchronization']) * abs(data['lead_lag_relationship'])
    
    # Regime-Adaptive Weighting
    data['efficiency_regime'] = (
        (data['price_discovery_efficiency'] > 0.8) & 
        (data['volume_clustering_efficiency'] < 0.3)
    ).astype(int)
    
    data['inefficiency_regime'] = (
        (data['information_absorption_rate'] < 0.1) & 
        (data['regime_persistence'] > 0.7)
    ).astype(int)
    
    data['transition_momentum'] = (
        data['net_pressure_accumulation'] * data['depth_resilience'] * data['volume_persistence']
    )
    
    # Multi-Timeframe Factor Components
    data['short_term_component'] = data['efficiency_divergence'] * data['net_pressure_accumulation']
    data['medium_term_component'] = data['efficiency_persistence'] * data['volume_persistence']
    data['structural_component'] = data['coupling_strength'] * data['transition_momentum']
    
    # Adaptive Factor Synthesis
    # Efficiency-Driven Factor
    data['efficiency_base'] = data['efficiency_divergence'] * data['net_pressure_accumulation']
    data['efficiency_enhanced'] = data['efficiency_base'] * (1 + data['volume_persistence'] / 5)
    data['efficiency_driven_factor'] = np.where(
        data['efficiency_regime'] == 1,
        data['efficiency_enhanced'] * data['price_discovery_efficiency'],
        data['efficiency_enhanced']
    )
    
    # Inefficiency Exploitation Factor
    data['inefficiency_base'] = data['efficiency_persistence'] * data['volume_persistence']
    data['inefficiency_enhanced'] = data['inefficiency_base'] * abs(data['information_absorption_rate'])
    data['inefficiency_exploitation_factor'] = np.where(
        data['inefficiency_regime'] == 1,
        data['inefficiency_enhanced'] * data['effective_spread_depth'],
        data['inefficiency_enhanced']
    )
    
    # Hierarchical Range-Volume Efficiency Factor
    data['core_factor'] = data['efficiency_driven_factor'] + data['inefficiency_exploitation_factor']
    data['structural_adjustment'] = data['core_factor'] * (1 + data['transition_momentum'])
    data['final_factor'] = data['structural_adjustment'] * data['coupling_strength']
    
    # Return the final factor series
    return data['final_factor']
