import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility Regime Adaptive Momentum factor that combines:
    - Volatility regime identification
    - Regime-specific factor construction
    - Adaptive signal integration
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate daily returns
    data['returns'] = data['close'].pct_change()
    
    # 1. Regime Identification
    # Calculate 20-day volatility (std of Close returns)
    data['volatility_20d'] = data['returns'].rolling(window=20).std()
    
    # Compute 5-day volatility momentum (change in volatility)
    data['vol_momentum_5d'] = data['volatility_20d'].pct_change(periods=5)
    
    # Classify regimes
    vol_median = data['volatility_20d'].rolling(window=60).median()
    data['high_vol_regime'] = data['volatility_20d'] > vol_median
    data['increasing_vol'] = data['vol_momentum_5d'] > 0
    
    # 2. Regime-Specific Factor Construction
    
    # High volatility regime: Price-Volume Divergence Factor
    data['price_momentum_3d'] = data['close'] / data['close'].shift(3)
    data['volume_momentum_3d'] = data['volume'] / data['volume'].shift(3)
    data['price_volume_divergence'] = data['price_momentum_3d'] - data['volume_momentum_3d']
    
    # Low volatility regime: Range Efficiency Factor
    data['daily_range'] = data['high'] - data['low']
    data['range_efficiency'] = (data['close'] - data['low']) / np.where(data['daily_range'] == 0, 1, data['daily_range'])
    data['range_efficiency_trend'] = data['range_efficiency'].rolling(window=5).mean()
    
    # Transition regime: Adaptive Momentum Factor
    data['vol_adjusted_returns'] = data['returns'] / np.where(data['volatility_20d'] == 0, 1, data['volatility_20d'])
    data['momentum_persistence'] = data['returns'].rolling(window=5).apply(
        lambda x: np.sum(np.sign(x) == np.sign(x.iloc[-1])) if len(x) == 5 else np.nan
    )
    
    # 3. Signal Integration
    factor_values = []
    
    for i in range(len(data)):
        if i < 20:  # Need enough data for volatility calculation
            factor_values.append(0)
            continue
            
        current_data = data.iloc[i]
        
        if current_data['high_vol_regime']:
            # High volatility regime - use price-volume divergence
            if not pd.isna(current_data['price_volume_divergence']):
                factor_values.append(current_data['price_volume_divergence'])
            else:
                factor_values.append(0)
                
        elif not current_data['high_vol_regime'] and not current_data['increasing_vol']:
            # Low volatility regime - use range efficiency trend
            if not pd.isna(current_data['range_efficiency_trend']):
                factor_values.append(current_data['range_efficiency_trend'])
            else:
                factor_values.append(0)
                
        else:
            # Transition regime - use adaptive momentum
            vol_adj_signal = current_data['vol_adjusted_returns'] if not pd.isna(current_data['vol_adjusted_returns']) else 0
            persistence_signal = current_data['momentum_persistence'] if not pd.isna(current_data['momentum_persistence']) else 0
            transition_signal = vol_adj_signal * persistence_signal
            factor_values.append(transition_signal)
    
    # Create factor series
    factor = pd.Series(factor_values, index=data.index)
    
    # Normalize the factor
    factor = (factor - factor.rolling(window=60).mean()) / factor.rolling(window=60).std()
    
    return factor.fillna(0)
