import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Aware Momentum-Volume Divergence Factor
    Combines price and volume momentum with amount-based regime detection
    """
    # Create copy to avoid modifying original data
    data = df.copy()
    
    # Momentum Calculation
    # Price Momentum
    data['price_momentum_5'] = data['close'] / data['close'].shift(5) - 1
    data['price_momentum_10'] = data['close'] / data['close'].shift(10) - 1
    data['price_momentum_20'] = data['close'] / data['close'].shift(20) - 1
    
    # Volume Momentum
    data['volume_momentum_5'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_momentum_10'] = data['volume'] / data['volume'].shift(10) - 1
    data['volume_momentum_20'] = data['volume'] / data['volume'].shift(20) - 1
    
    # Regime Detection using Amount Data
    data['amount_trend_10'] = data['amount'] / data['amount'].shift(10)
    data['amount_acceleration'] = data['amount_trend_10'] - data['amount_trend_10'].shift(1)
    
    # Regime Classification
    def classify_regime(row):
        if row['amount_trend_10'] > 1.05 and row['amount_acceleration'] > 0:
            return 2  # Bullish
        elif row['amount_trend_10'] < 0.95 and row['amount_acceleration'] < 0:
            return 0  # Bearish
        else:
            return 1  # Transition
    
    data['regime'] = data.apply(classify_regime, axis=1)
    
    # Exponential Smoothing Application
    alpha = 0.3
    
    # EMA for price momentum
    data['ema_price_5'] = data['price_momentum_5'].ewm(alpha=alpha).mean()
    data['ema_price_10'] = data['price_momentum_10'].ewm(alpha=alpha).mean()
    data['ema_price_20'] = data['price_momentum_20'].ewm(alpha=alpha).mean()
    
    # EMA for volume momentum
    data['ema_volume_5'] = data['volume_momentum_5'].ewm(alpha=alpha).mean()
    data['ema_volume_10'] = data['volume_momentum_10'].ewm(alpha=alpha).mean()
    data['ema_volume_20'] = data['volume_momentum_20'].ewm(alpha=alpha).mean()
    
    # Momentum acceleration
    data['price_accel_5'] = data['ema_price_5'] - data['ema_price_5'].shift(1)
    data['volume_accel_5'] = data['ema_volume_5'] - data['ema_volume_5'].shift(1)
    
    # Divergence Pattern Analysis
    # Directional divergence (sign mismatch)
    data['directional_div_5'] = np.sign(data['ema_price_5']) != np.sign(data['ema_volume_5'])
    data['directional_div_10'] = np.sign(data['ema_price_10']) != np.sign(data['ema_volume_10'])
    data['directional_div_20'] = np.sign(data['ema_price_20']) != np.sign(data['ema_volume_20'])
    
    # Magnitude divergence (relative strength)
    data['magnitude_div_5'] = (data['ema_price_5'] - data['ema_volume_5']) / (np.abs(data['ema_price_5']) + np.abs(data['ema_volume_5']) + 1e-8)
    data['magnitude_div_10'] = (data['ema_price_10'] - data['ema_volume_10']) / (np.abs(data['ema_price_10']) + np.abs(data['ema_volume_10']) + 1e-8)
    data['magnitude_div_20'] = (data['ema_price_20'] - data['ema_volume_20']) / (np.abs(data['ema_price_20']) + np.abs(data['ema_volume_20']) + 1e-8)
    
    # Multi-timeframe consistency
    data['consistency_score'] = (
        data['directional_div_5'].astype(int) + 
        data['directional_div_10'].astype(int) + 
        data['directional_div_20'].astype(int)
    ) / 3.0
    
    # Acceleration discrepancy
    data['accel_discrepancy'] = data['price_accel_5'] - data['volume_accel_5']
    
    # Combine divergence metrics with regime context
    def calculate_divergence_strength(row):
        base_divergence = (
            row['magnitude_div_5'] * 0.5 + 
            row['magnitude_div_10'] * 0.3 + 
            row['magnitude_div_20'] * 0.2
        )
        
        # Adjust for regime
        if row['regime'] == 2:  # Bullish regime
            regime_multiplier = 1.2
        elif row['regime'] == 0:  # Bearish regime
            regime_multiplier = 0.8
        else:  # Transition regime
            regime_multiplier = 1.0
            
        # Incorporate acceleration and consistency
        final_score = (
            base_divergence * regime_multiplier + 
            row['accel_discrepancy'] * 0.3 + 
            row['consistency_score'] * 0.2
        )
        
        return final_score
    
    data['divergence_strength'] = data.apply(calculate_divergence_strength, axis=1)
    
    # Cross-sectional ranking (within the available universe)
    data['factor_rank'] = data['divergence_strength'].rank(pct=True)
    
    # Signal generation with regime-aware thresholds
    def generate_signal(row):
        if row['regime'] == 2:  # Bullish regime
            if row['factor_rank'] > 0.8 and row['ema_price_5'] > 0:
                return 1  # Strong bullish
            elif row['factor_rank'] < 0.2 and row['ema_price_5'] < 0:
                return -1  # Strong bearish
        elif row['regime'] == 0:  # Bearish regime
            if row['factor_rank'] > 0.8 and row['ema_price_5'] < 0:
                return -1  # Strong bearish
            elif row['factor_rank'] < 0.2 and row['ema_price_5'] > 0:
                return 1  # Strong bullish (contrarian)
        else:  # Transition regime
            if row['factor_rank'] > 0.8:
                return 0.5  # Moderate bullish
            elif row['factor_rank'] < 0.2:
                return -0.5  # Moderate bearish
        
        return 0  # Neutral
    
    data['factor_signal'] = data.apply(generate_signal, axis=1)
    
    # Final factor output with stationarity consideration
    # Apply rolling standardization to maintain stationarity
    factor_series = data['factor_signal'].rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-8) if x.std() > 0 else 0
    )
    
    return factor_series
