import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Fractal Cross-Asset Momentum with Volume-Efficiency Regime Dynamics
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Helper function to calculate fractal dimension using box-counting method
    def calculate_fractal_dimension(series, window=5):
        """Calculate fractal dimension using simplified box-counting method"""
        fractal_dims = []
        for i in range(len(series)):
            if i < window - 1:
                fractal_dims.append(np.nan)
                continue
                
            window_data = series.iloc[i-window+1:i+1]
            if len(window_data) < window:
                fractal_dims.append(np.nan)
                continue
                
            # Simplified box-counting method
            range_val = window_data.max() - window_data.min()
            if range_val == 0:
                fractal_dims.append(1.0)
                continue
                
            # Calculate number of boxes needed (simplified)
            std_val = window_data.std()
            if std_val == 0:
                fractal_dims.append(1.0)
                continue
                
            # Fractal dimension approximation
            fractal_dim = 2 - (np.log(std_val) / np.log(range_val)) if range_val > 0 and std_val > 0 else 1.0
            fractal_dims.append(max(1.0, min(2.0, fractal_dim)))
            
        return pd.Series(fractal_dims, index=series.index)
    
    # Calculate price fractal dimensions
    data['price_fractal_5d'] = calculate_fractal_dimension(data['close'], 5)
    data['price_fractal_3d'] = calculate_fractal_dimension(data['close'], 3)
    data['price_fractal_8d'] = calculate_fractal_dimension(data['close'], 8)
    
    # Calculate volume fractal dimensions
    data['volume_fractal_3d'] = calculate_fractal_dimension(data['volume'], 3)
    data['volume_fractal_5d'] = calculate_fractal_dimension(data['volume'], 5)
    
    # Calculate fractal changes and accelerations
    data['price_fractal_change_3d'] = data['price_fractal_3d'] - data['price_fractal_3d'].shift(3)
    data['price_fractal_change_1d'] = data['price_fractal_3d'] - data['price_fractal_3d'].shift(1)
    data['price_fractal_acceleration'] = data['price_fractal_change_1d'] - data['price_fractal_change_1d'].shift(1)
    
    data['volume_fractal_change_3d'] = data['volume_fractal_3d'] - data['volume_fractal_3d'].shift(3)
    data['volume_fractal_change_1d'] = data['volume_fractal_3d'] - data['volume_fractal_3d'].shift(1)
    data['volume_fractal_acceleration'] = data['volume_fractal_change_1d'] - data['volume_fractal_change_1d'].shift(1)
    
    # Volume-Price Fractal Alignment
    data['volume_price_fractal_alignment'] = data['volume_fractal_change_3d'] * data['price_fractal_change_3d']
    
    # Amount Flow Fractal Analysis
    data['price_change'] = data['close'] - data['close'].shift(1)
    data['positive_flow'] = np.where(data['price_change'] > 0, data['amount'], 0)
    data['negative_flow'] = np.where(data['price_change'] < 0, data['amount'], 0)
    
    # Calculate positive and negative flow fractals
    positive_flow_rolling = data['positive_flow'].rolling(window=5, min_periods=3)
    negative_flow_rolling = data['negative_flow'].rolling(window=5, min_periods=3)
    amount_rolling = data['amount'].rolling(window=5, min_periods=3)
    
    data['positive_flow_fractal'] = np.log(5) / np.log(
        positive_flow_rolling.sum() / (amount_rolling.max() - amount_rolling.min() + 1e-8)
    )
    data['negative_flow_fractal'] = np.log(5) / np.log(
        negative_flow_rolling.sum() / (amount_rolling.max() - amount_rolling.min() + 1e-8)
    )
    data['flow_fractal_imbalance'] = data['positive_flow_fractal'] - data['negative_flow_fractal']
    
    # Range Efficiency Fractal Analysis
    data['daily_range'] = data['high'] - data['low']
    data['intraday_efficiency'] = (data['close'] - data['low']) / (data['daily_range'] + 1e-8)
    data['intraday_fractal_efficiency'] = data['intraday_efficiency'] * data['price_fractal_5d']
    
    # Multi-day range fractal
    range_rolling = data['daily_range'].rolling(window=5, min_periods=3)
    data['range_fractal'] = np.log(5) / np.log(
        range_rolling.sum() / (data['high'].rolling(5).max() - data['low'].rolling(5).min() + 1e-8)
    )
    data['efficiency_fractal_convergence'] = data['range_fractal'] * data['intraday_fractal_efficiency']
    
    # Asymmetric Fractal Momentum Components
    data['upside_fractal_momentum'] = 0.0
    data['downside_fractal_momentum'] = 0.0
    
    for i in range(1, 6):
        up_mask = data['close'] > data['close'].shift(i)
        down_mask = data['close'] < data['close'].shift(i)
        
        fractal_change = data['price_fractal_3d'] - data['price_fractal_3d'].shift(i)
        data['upside_fractal_momentum'] += np.where(up_mask, fractal_change, 0)
        data['downside_fractal_momentum'] += np.where(down_mask, fractal_change, 0)
    
    # Normalize by volatility
    upside_vol = data['close'].pct_change().where(data['close'] > data['close'].shift(1)).rolling(5).std()
    downside_vol = data['close'].pct_change().where(data['close'] < data['close'].shift(1)).rolling(5).std()
    
    data['upside_fractal_momentum'] = data['upside_fractal_momentum'] / (upside_vol + 1e-8)
    data['downside_fractal_momentum'] = data['downside_fractal_momentum'] / (downside_vol + 1e-8)
    
    data['fractal_momentum_asymmetry'] = data['upside_fractal_momentum'] / (
        data['upside_fractal_momentum'] + np.abs(data['downside_fractal_momentum']) + 1e-8
    )
    
    # Fractal Position Analysis
    data['intraday_fractal_position'] = (
        data['close'] - data['low'].rolling(window=3).min()
    ) / (data['high'].rolling(window=3).max() - data['low'].rolling(window=3).min() + 1e-8)
    
    data['fractal_position_momentum'] = (
        data['intraday_fractal_position'] - data['intraday_fractal_position'].shift(1)
    ) * data['price_fractal_change_1d']
    
    # Fractal Efficiency-Position Score
    data['fractal_efficiency_position_score'] = (
        data['volume_fractal_3d'] * data['intraday_fractal_position'] * data['efficiency_fractal_convergence']
    )
    
    # Flow-Position Fractal Score
    data['flow_position_fractal_score'] = data['flow_fractal_imbalance'] * data['intraday_fractal_position']
    
    # Regime Classification
    data['volume_efficiency_regime'] = 0  # Default: Stable regime
    
    # High Efficiency Regime
    high_efficiency_mask = (
        (data['volume_price_fractal_alignment'] > data['volume_price_fractal_alignment'].rolling(20).quantile(0.7)) &
        (np.abs(data['flow_fractal_imbalance']) > data['flow_fractal_imbalance'].rolling(20).std())
    )
    data.loc[high_efficiency_mask, 'volume_efficiency_regime'] = 1
    
    # Low Efficiency Regime
    low_efficiency_mask = (
        (data['volume_price_fractal_alignment'] < data['volume_price_fractal_alignment'].rolling(20).quantile(0.3)) &
        (np.abs(data['flow_fractal_imbalance']) < data['flow_fractal_imbalance'].rolling(20).std() * 0.5)
    )
    data.loc[low_efficiency_mask, 'volume_efficiency_regime'] = 2
    
    # Compression Regime
    compression_mask = (
        (data['range_fractal'] < data['range_fractal'].rolling(20).quantile(0.3)) &
        (np.abs(data['efficiency_fractal_convergence']) > data['efficiency_fractal_convergence'].rolling(20).quantile(0.4))
    )
    data.loc[compression_mask, 'volume_efficiency_regime'] = 3
    
    # Expansion Regime
    expansion_mask = (
        (data['range_fractal'] > data['range_fractal'].rolling(20).quantile(0.7)) &
        (data['efficiency_fractal_convergence'].rolling(5).std() > data['efficiency_fractal_convergence'].rolling(20).std())
    )
    data.loc[expansion_mask, 'volume_efficiency_regime'] = 4
    
    # Composite Alpha Calculation with Regime-Adaptive Weighting
    # Primary Factor: Cross-Asset Fractal Momentum × Volume-Fractal Efficiency
    primary_factor = data['price_fractal_acceleration'] * data['volume_price_fractal_alignment']
    
    # Confirmation Factor: Flow Fractal Position Alignment × Efficiency-Fractal Convergence
    confirmation_factor = data['flow_position_fractal_score'] * data['efficiency_fractal_convergence']
    
    # Apply regime-specific multipliers
    regime_multipliers = {
        1: 1.2,  # High Efficiency: Emphasize cross-asset momentum
        2: 0.8,  # Low Efficiency: Conservative weighting
        3: 1.5,  # Compression: Amplify signals
        4: 0.9   # Expansion: Reduce weighting due to volatility
    }
    
    # Calculate regime-adjusted factors
    regime_multiplier = data['volume_efficiency_regime'].map(regime_multipliers).fillna(1.0)
    
    # Final composite alpha
    composite_alpha = (
        primary_factor * regime_multiplier * 0.6 +
        confirmation_factor * regime_multiplier * 0.4 +
        data['fractal_momentum_asymmetry'] * 0.2 +
        data['fractal_efficiency_position_score'] * 0.1
    )
    
    # Apply cross-validation using fractal correlation proxy
    fractal_consistency = 1.0 - np.abs(data['price_fractal_5d'] - data['price_fractal_3d'])
    final_alpha = composite_alpha * fractal_consistency
    
    # Normalize and clean
    final_alpha = final_alpha.replace([np.inf, -np.inf], np.nan)
    final_alpha = (final_alpha - final_alpha.rolling(50, min_periods=20).mean()) / (
        final_alpha.rolling(50, min_periods=20).std() + 1e-8
    )
    
    return final_alpha
