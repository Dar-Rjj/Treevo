import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Regime Identification
    # Volatility regime via rolling True Range
    data['tr'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['volatility_regime'] = data['tr'].rolling(window=20).mean()
    volatility_threshold = data['volatility_regime'].rolling(window=60).quantile(0.6)
    
    # Trending vs ranging via price fractal dimension approximation
    def calculate_fractal_dimension(high, low, window=20):
        fractal_dim = pd.Series(index=high.index, dtype=float)
        for i in range(window, len(high)):
            window_high = high.iloc[i-window:i]
            window_low = low.iloc[i-window:i]
            price_range = window_high.max() - window_low.min()
            if price_range > 0:
                # Simplified fractal dimension approximation
                daily_ranges = window_high.values - window_low.values
                total_path = np.sum(daily_ranges)
                fractal_dim.iloc[i] = np.log(total_path) / np.log(price_range)
            else:
                fractal_dim.iloc[i] = 1.0
        return fractal_dim
    
    data['fractal_dim'] = calculate_fractal_dimension(data['high'], data['low'])
    trend_threshold = data['fractal_dim'].rolling(window=60).quantile(0.4)
    
    # 2. Momentum Component
    # Multi-window returns
    data['ret_5'] = data['close'].pct_change(5)
    data['ret_10'] = data['close'].pct_change(10)
    data['ret_20'] = data['close'].pct_change(20)
    
    # Momentum acceleration via rate of change
    data['momentum_accel'] = (data['ret_5'] - data['ret_10'].shift(5)) / 5
    
    # Combined momentum score
    data['momentum_score'] = (
        0.4 * data['ret_5'] + 
        0.3 * data['ret_10'] + 
        0.3 * data['ret_20'] +
        0.2 * data['momentum_accel']
    )
    
    # 3. Divergence Detection
    # Price-volume trend correlation
    data['price_trend'] = data['close'].rolling(window=10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True
    )
    data['volume_trend'] = data['volume'].rolling(window=10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True
    )
    
    # Divergence strength measurement
    data['pv_divergence'] = (
        np.sign(data['price_trend']) * np.sign(data['volume_trend']) * 
        (abs(data['price_trend']) - abs(data['volume_trend']))
    )
    
    # 4. Signal Generation
    factor = pd.Series(index=data.index, dtype=float)
    
    for i in range(20, len(data)):
        current_vol = data['volatility_regime'].iloc[i]
        vol_threshold = volatility_threshold.iloc[i]
        current_fractal = data['fractal_dim'].iloc[i]
        trend_thresh = trend_threshold.iloc[i]
        
        # High volatility regime
        if current_vol > vol_threshold:
            # Mean reversion with volume confirmation
            recent_returns = data['ret_5'].iloc[i-4:i+1].mean() if i >= 4 else 0
            volume_confirmation = data['volume'].iloc[i] > data['volume'].rolling(window=10).mean().iloc[i]
            
            if abs(recent_returns) > 0.02:  # Significant move
                if recent_returns > 0 and volume_confirmation:
                    factor.iloc[i] = -data['momentum_score'].iloc[i] * (1 + abs(data['pv_divergence'].iloc[i]))
                elif recent_returns < 0 and volume_confirmation:
                    factor.iloc[i] = data['momentum_score'].iloc[i] * (1 + abs(data['pv_divergence'].iloc[i]))
                else:
                    factor.iloc[i] = -data['momentum_score'].iloc[i] * 0.5
            else:
                factor.iloc[i] = data['momentum_score'].iloc[i] * 0.3
        
        # Low volatility regime
        else:
            # Momentum persistence with trend strength
            if current_fractal < trend_thresh:  # Trending market
                trend_strength = abs(data['price_trend'].iloc[i]) / data['close'].iloc[i]
                factor.iloc[i] = data['momentum_score'].iloc[i] * (1 + trend_strength)
            else:  # Ranging market
                # Use divergence as contrarian signal
                if abs(data['pv_divergence'].iloc[i]) > 0.001:
                    factor.iloc[i] = -np.sign(data['pv_divergence'].iloc[i]) * data['momentum_score'].iloc[i]
                else:
                    factor.iloc[i] = data['momentum_score'].iloc[i] * 0.5
    
    # Fill early NaN values with 0
    factor = factor.fillna(0)
    
    return factor
