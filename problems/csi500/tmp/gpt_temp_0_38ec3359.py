import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Momentum-Accelerated Liquidity Absorption alpha factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Bidirectional Liquidity Flow
    # High-Low Liquidity Capture
    data['liquidity_efficiency'] = (data['high'] - data['low']) / (data['amount'] + 1e-8)
    data['liquidity_efficiency_momentum'] = data['liquidity_efficiency'] / data['liquidity_efficiency'].shift(3) - 1
    
    # Volume Absorption Patterns
    data['volume_ma_5'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_concentration'] = data['volume'] / (data['volume_ma_5'] + 1e-8)
    
    # Absorption detection: high volume with small price range
    price_range = (data['high'] - data['low']) / data['close']
    data['absorption_detected'] = ((data['volume_concentration'] > 1.2) & 
                                  (price_range < price_range.rolling(window=10, min_periods=5).mean())).astype(int)
    
    # Absorption intensity score
    data['absorption_intensity'] = (data['volume_concentration'] * 
                                   (1 - price_range / price_range.rolling(window=10, min_periods=5).mean()))
    
    # 2. Momentum Acceleration Dynamics
    # Multi-timeframe Momentum
    data['momentum_short'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_medium'] = data['close'] / data['close'].shift(10) - 1
    data['momentum_acceleration'] = data['momentum_short'] / (np.abs(data['momentum_medium']) + 1e-8)
    
    # Momentum Regime Shifts
    momentum_change = data['momentum_short'].diff()
    data['momentum_inflection'] = ((momentum_change * momentum_change.shift(1)) < 0).astype(int)
    
    # Momentum persistence score
    data['momentum_persistence'] = (data['momentum_short'].rolling(window=5, min_periods=3)
                                   .apply(lambda x: len([i for i in range(1, len(x)) if x[i] * x[i-1] > 0]) / max(1, len(x)-1)))
    
    # Momentum exhaustion signals
    momentum_volatility = data['momentum_short'].rolling(window=10, min_periods=5).std()
    data['momentum_exhaustion'] = (np.abs(data['momentum_short']) > 2 * momentum_volatility).astype(int)
    
    # 3. Liquidity-Momentum Divergence
    # Liquidity Efficiency vs Momentum Divergence
    liquidity_trend = data['liquidity_efficiency'].rolling(window=5, min_periods=3).mean()
    momentum_direction = np.sign(data['momentum_short'])
    data['liquidity_momentum_divergence'] = ((liquidity_trend.diff() * momentum_direction) < 0).astype(int)
    
    # Absorption-Momentum Timing Patterns
    data['absorption_acceleration'] = data['absorption_intensity'] * data['momentum_acceleration']
    data['absorption_extreme'] = ((data['absorption_intensity'] > data['absorption_intensity'].rolling(window=10, min_periods=5).quantile(0.8)) & 
                                 (np.abs(data['momentum_short']) > data['momentum_short'].rolling(window=10, min_periods=5).std())).astype(int)
    
    # Timing efficiency score
    data['timing_efficiency'] = (data['absorption_acceleration'] * 
                                (1 - data['liquidity_momentum_divergence']) * 
                                (1 - data['momentum_exhaustion']))
    
    # 4. Composite Alpha Signal
    # Liquidity-Momentum Convergence Score
    convergence_base = data['absorption_intensity'] * data['momentum_acceleration']
    divergence_adjustment = 1 - 0.5 * data['liquidity_momentum_divergence'] - 0.3 * data['momentum_exhaustion']
    data['liquidity_momentum_score'] = convergence_base * divergence_adjustment
    
    # Dynamic Signal Thresholding
    liquidity_regime = data['volume_concentration'].rolling(window=10, min_periods=5).mean()
    momentum_phase = data['momentum_short'].rolling(window=5, min_periods=3).mean()
    
    # Momentum-phase specific thresholds
    high_momentum_threshold = data['liquidity_momentum_score'].rolling(window=20, min_periods=10).quantile(0.7)
    low_momentum_threshold = data['liquidity_momentum_score'].rolling(window=20, min_periods=10).quantile(0.3)
    
    # Multi-timeframe confirmation
    short_confirmation = data['liquidity_momentum_score'] > data['liquidity_momentum_score'].shift(3)
    medium_confirmation = data['liquidity_momentum_score'] > data['liquidity_momentum_score'].shift(5)
    
    # Final alpha signal
    data['alpha_signal'] = (data['liquidity_momentum_score'] * 
                           ((np.abs(momentum_phase) > momentum_phase.rolling(window=20, min_periods=10).std()) + 1) *
                           (short_confirmation.astype(int) + medium_confirmation.astype(int)) / 2)
    
    # Apply dynamic thresholding based on momentum phase
    momentum_condition = np.abs(momentum_phase) > momentum_phase.rolling(window=20, min_periods=10).std()
    data['alpha_signal'] = np.where(momentum_condition, 
                                   data['alpha_signal'] * (data['liquidity_momentum_score'] > high_momentum_threshold),
                                   data['alpha_signal'] * (data['liquidity_momentum_score'] > low_momentum_threshold))
    
    # Clean up intermediate columns and return final signal
    result = data['alpha_signal'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return result
