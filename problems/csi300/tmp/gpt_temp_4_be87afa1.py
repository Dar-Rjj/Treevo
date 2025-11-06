import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Dynamic Regime-Adaptive Momentum Divergence alpha factor
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Momentum Regime Classification
    # Multi-Timeframe Momentum Assessment
    data['momentum_3d'] = data['close'].pct_change(3)
    data['momentum_10d'] = data['close'].pct_change(10)
    data['momentum_20d'] = data['close'].pct_change(20)
    
    # Momentum Acceleration Analysis
    data['accel_short'] = data['momentum_3d'] - data['momentum_10d'].shift(3)
    data['accel_medium'] = data['momentum_10d'] - data['momentum_20d'].shift(10)
    
    # Regime Classification
    conditions = [
        (data['momentum_20d'] > 0) & (data['accel_short'] > 0),  # Bullish accelerating
        (data['momentum_20d'] > 0) & (data['accel_short'] <= 0),  # Bullish decelerating
        (data['momentum_20d'] <= 0) & (data['accel_short'] > 0),  # Bearish accelerating
        (data['momentum_20d'] <= 0) & (data['accel_short'] <= 0)  # Bearish decelerating
    ]
    choices = [2, 1, -1, -2]  # Higher absolute values for accelerating regimes
    data['momentum_regime'] = np.select(conditions, choices, default=0)
    
    # Volatility Regime Analysis
    # True Range Calculation
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['close'].shift(1))
    data['tr3'] = abs(data['low'] - data['close'].shift(1))
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    data['volatility_5d'] = data['true_range'].rolling(window=5).std()
    
    # Volatility Persistence
    data['vol_autocorr'] = data['volatility_5d'].rolling(window=5).apply(
        lambda x: x.autocorr(), raw=False
    ).fillna(0)
    
    # Volatility Regime Detection
    vol_median = data['volatility_5d'].rolling(window=20).median()
    data['vol_regime'] = np.where(
        data['volatility_5d'] > vol_median * 1.2, 2,  # High volatility
        np.where(data['volatility_5d'] < vol_median * 0.8, 0.5, 1)  # Low volatility / Normal
    )
    
    # Volume-Price Divergence Detection
    # Volume Pattern Analysis
    data['volume_20d_avg'] = data['volume'].rolling(window=20).mean()
    data['volume_abnormal'] = data['volume'] / data['volume_20d_avg']
    data['volume_trend'] = data['volume'].rolling(window=5).apply(
        lambda x: np.polyfit(range(5), x, 1)[0], raw=False
    ).fillna(0)
    
    # Divergence Signal Generation
    data['price_change_1d'] = data['close'].pct_change()
    
    # Positive Divergence (accumulation)
    pos_div_cond1 = (data['price_change_1d'] < 0) & (data['volume_abnormal'] > 1.5)
    pos_div_cond2 = (abs(data['price_change_1d']) < 0.005) & (data['volume_abnormal'] > 2.0)
    data['pos_divergence'] = np.where(pos_div_cond1 | pos_div_cond2, 1, 0)
    
    # Negative Divergence (distribution)
    neg_div_cond1 = (data['price_change_1d'] > 0) & (data['volume_abnormal'] < 0.7)
    neg_div_cond2 = (abs(data['price_change_1d']) > 0.02) & (data['volume_abnormal'] < 0.8)
    data['neg_divergence'] = np.where(neg_div_cond1 | neg_div_cond2, -1, 0)
    
    # Combined divergence signal
    data['divergence_signal'] = data['pos_divergence'] + data['neg_divergence']
    
    # Divergence confidence (persistence)
    data['divergence_strength'] = data['divergence_signal'].rolling(window=3).sum()
    
    # Adaptive Factor Combination
    # Regime-Dependent Weighting
    momentum_weight = np.where(abs(data['momentum_regime']) == 2, 1.5, 1.0)  # Higher for accelerating
    
    # Volatility regime adjustments
    volatility_weight = 1.0 / data['vol_regime']  # Lower weight in high volatility
    
    # Divergence confidence weighting
    divergence_weight = np.where(abs(data['divergence_strength']) >= 2, 1.5, 
                                np.where(abs(data['divergence_strength']) == 1, 1.0, 0.5))
    
    # Composite Alpha Generation
    # Base momentum signal (weighted by regime)
    base_momentum = data['momentum_10d'] * momentum_weight
    
    # Apply divergence signals
    momentum_divergence = base_momentum * data['divergence_signal'] * divergence_weight
    
    # Final volatility scaling
    final_alpha = momentum_divergence * volatility_weight
    
    # Dynamic adjustment based on recent performance
    alpha_performance = final_alpha.rolling(window=5).mean()
    adaptive_weight = 1 + np.tanh(alpha_performance * 10)  # Scale based on recent success
    
    # Final composite alpha
    composite_alpha = final_alpha * adaptive_weight
    
    # Clean up and return
    result = pd.Series(composite_alpha, index=data.index)
    result = result.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)
    
    return result
