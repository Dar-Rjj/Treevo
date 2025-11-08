import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate Momentum Components
    # Short-Term Momentum
    data['momentum_5d'] = data['close'].pct_change(5)
    data['momentum_10d'] = data['close'].pct_change(10)
    
    # Momentum Acceleration
    data['momentum_accel_5d'] = data['momentum_5d'].diff()
    data['momentum_accel_10d'] = data['momentum_10d'].diff()
    
    # Analyze Volume Dynamics
    # Volume Trend Components
    data['volume_ema_10d'] = data['volume'].ewm(span=10, adjust=False).mean()
    data['volume_momentum_5d'] = data['volume'].pct_change(5)
    
    # Volume Acceleration
    data['volume_momentum_1d'] = data['volume'].pct_change()
    data['volume_accel_1d'] = data['volume_momentum_1d'].diff()
    data['volume_accel_5d'] = data['volume_momentum_5d'].diff()
    
    # Calculate Intraday Efficiency
    # Daily Price Efficiency
    data['daily_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['efficiency_5d_avg'] = data['daily_efficiency'].rolling(window=5).mean()
    
    # Efficiency Momentum
    data['efficiency_momentum_3d'] = data['daily_efficiency'].diff(3)
    data['efficiency_weight'] = data['daily_efficiency'].rolling(window=5).mean()
    
    # Detect Divergence Patterns
    # Price-Volume-Efficiency Direction Analysis
    data['price_direction'] = np.sign(data['momentum_accel_5d'])
    data['volume_direction'] = np.sign(data['volume_accel_5d'])
    data['efficiency_direction'] = np.sign(data['efficiency_momentum_3d'])
    
    # Quantify Divergence Strength
    data['price_volume_divergence'] = data['momentum_accel_5d'] * data['volume_accel_5d']
    data['divergence_strength'] = data['price_volume_divergence'] * data['efficiency_momentum_3d']
    
    # Direction agreement factor
    data['direction_agreement'] = (data['price_direction'] * data['volume_direction'] * 
                                  data['efficiency_direction'])
    data['divergence_signed'] = data['divergence_strength'] * data['direction_agreement']
    
    # Incorporate Volatility Context
    # Calculate Rolling Volatility
    data['daily_range'] = data['high'] - data['low']
    data['range_volatility_20d'] = data['daily_range'].rolling(window=20).std()
    data['price_volatility_20d'] = data['close'].pct_change().rolling(window=20).std()
    
    # Adjust Signal by Volatility Regime
    data['volatility_combined'] = (data['range_volatility_20d'] + data['price_volatility_20d']) / 2
    data['volatility_scaling'] = 1 / (1 + data['volatility_combined'])
    
    # Synthesize Composite Factor
    # Combine All Components
    data['efficiency_adjusted_momentum'] = data['momentum_5d'] * (1 + data['efficiency_weight'])
    data['core_factor'] = data['divergence_signed'] * data['efficiency_adjusted_momentum']
    data['volatility_scaled_factor'] = data['core_factor'] * data['volatility_scaling']
    
    # Apply Liquidity Validation
    data['volume_rank'] = data['volume'].rolling(window=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    data['amount_per_volume'] = data['amount'] / (data['volume'] + 1e-8)
    data['amount_consistency'] = data['amount_per_volume'].rolling(window=5).std()
    
    # Liquidity filter
    data['liquidity_filter'] = np.where(
        (data['volume_rank'] > 0.3) & (data['amount_consistency'] < data['amount_consistency'].rolling(20).quantile(0.8)),
        1.0, 0.5
    )
    
    # Final Signal Construction
    data['final_factor'] = data['volatility_scaled_factor'] * data['liquidity_filter']
    
    # Apply bounds to prevent extremes
    factor_std = data['final_factor'].rolling(window=60).std()
    data['final_factor_bounded'] = data['final_factor'].clip(
        lower=-2.5 * factor_std, 
        upper=2.5 * factor_std
    )
    
    # Incorporate trend persistence
    data['trend_persistence'] = data['momentum_5d'].rolling(window=3).apply(
        lambda x: np.sum(np.sign(x) == np.sign(x.iloc[-1])) / len(x), raw=False
    )
    
    # Final factor with trend persistence enhancement
    data['enhanced_factor'] = data['final_factor_bounded'] * (1 + 0.2 * data['trend_persistence'])
    
    # Return the final factor series
    return data['enhanced_factor']
