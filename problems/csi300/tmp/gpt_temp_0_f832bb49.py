import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import entropy

def heuristics_v2(df):
    """
    Nonlinear Price-Volume Entropy with Microstructure Regime Detection
    """
    data = df.copy()
    
    # Helper function for entropy calculation
    def calculate_entropy(series, bins=10):
        hist, _ = np.histogram(series.dropna(), bins=bins)
        prob = hist / hist.sum()
        prob = prob[prob > 0]  # Remove zeros for log calculation
        return -np.sum(prob * np.log(prob))
    
    # Helper function for Hurst exponent estimation
    def hurst_exponent(ts):
        """Calculate Hurst exponent using rescaled range analysis"""
        if len(ts) < 10:
            return 0.5
        lags = range(2, min(10, len(ts)))
        tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    for i in range(20, len(data)):
        current_data = data.iloc[:i+1]
        
        # 1. Entropy-Based Price Efficiency
        # Shannon Entropy of Returns
        returns = current_data['close'].pct_change().dropna()
        if len(returns) >= 20:
            recent_returns = returns.tail(20)
            price_entropy = calculate_entropy(recent_returns)
            
            # Volume-Weighted Entropy
            vol_weighted_returns = recent_returns * current_data['volume'].tail(20).values
            vol_entropy = calculate_entropy(vol_weighted_returns)
            
            # Efficiency Ratio
            price_changes = np.abs(current_data['close'].diff().tail(20))
            net_change = np.abs(current_data['close'].iloc[-1] - current_data['close'].iloc[-21])
            efficiency = net_change / price_changes.sum() if price_changes.sum() > 0 else 0
            entropy_adjusted_eff = efficiency * (1 - price_entropy / np.log(10))
        
        # 2. Microstructure Regime Classification
        # Bid-Ask Spread Proxy
        mid_price = (current_data['high'] + current_data['low']) / 2
        effective_spread = 2 * np.abs(current_data['close'] - mid_price) / mid_price
        current_spread = effective_spread.iloc[-1]
        spread_momentum = current_spread / effective_spread.iloc[-2] if i > 20 else 1.0
        
        # Volume Clustering Patterns
        vol_ma = current_data['volume'].rolling(20).mean()
        volume_burst = current_data['volume'].iloc[-1] / vol_ma.iloc[-1] > 2 if not pd.isna(vol_ma.iloc[-1]) else False
        
        # Volume persistence
        above_ma = current_data['volume'].tail(5) > vol_ma.tail(5)
        volume_persistence = above_ma.sum() if len(above_ma) > 0 else 0
        
        # Volume entropy
        recent_volume = current_data['volume'].tail(20)
        volume_entropy = calculate_entropy(recent_volume)
        
        # Regime Classification
        high_friction = current_spread > 0.002 and volume_entropy < 0.5
        liquid = current_spread < 0.001 and volume_persistence > 3
        transitional = spread_momentum > 1.3 and volume_burst
        
        # 3. Nonlinear Price-Volume Coupling
        if len(current_data) >= 25:
            # Delayed Volume-Response Function
            volume_window = current_data['volume'].iloc[-6:-3]  # t-3:t
            return_window = current_data['close'].pct_change().iloc[-5:-2]  # t-2:t+1 (relative to volume)
            
            if len(volume_window) >= 3 and len(return_window) >= 3:
                vol_return_corr = volume_window.corr(return_window)
                
                # Asymmetric response (simplified)
                pos_returns = return_window[return_window > 0]
                neg_returns = return_window[return_window < 0]
                vol_pos = volume_window.iloc[:len(pos_returns)] if len(pos_returns) > 0 else pd.Series()
                vol_neg = volume_window.iloc[:len(neg_returns)] if len(neg_returns) > 0 else pd.Series()
                
                asym_response = (vol_pos.corr(pos_returns) if len(vol_pos) > 1 else 0) - \
                              (vol_neg.corr(neg_returns) if len(vol_neg) > 1 else 0)
            
            # Fractal Dimension (simplified)
            price_series = current_data['close'].tail(20).values
            volume_series = current_data['volume'].tail(20).values
            
            hurst_price = hurst_exponent(price_series)
            hurst_volume = hurst_exponent(volume_series)
            dimension_ratio = hurst_price / hurst_volume if hurst_volume != 0 else 1.0
            
            # Coupling Strength Metrics (simplified)
            price_changes = np.diff(price_series)
            volume_changes = np.diff(volume_series)
            if len(price_changes) > 1 and len(volume_changes) > 1:
                mutual_info = np.corrcoef(price_changes, volume_changes)[0,1]
                nonlinear_corr_ratio = abs(mutual_info)
            else:
                mutual_info = 0
                nonlinear_corr_ratio = 0
            
            # 4. Adaptive Signal Generation
            signal_strength = 0
            
            # High-Friction Mean Reversion
            if high_friction and price_entropy > 0.8 and efficiency < 0.3:
                signal_strength = (1 - efficiency) * price_entropy * abs(asym_response)
            
            # Liquid Trend Acceleration
            elif liquid and dimension_ratio > 1.2 and asym_response > 0:
                signal_strength = dimension_ratio * asym_response * abs(mutual_info)
            
            # Transitional Breakout Detection
            elif transitional and dimension_ratio > 1.1 and nonlinear_corr_ratio > 0.15:
                signal_strength = dimension_ratio * nonlinear_corr_ratio * abs(mutual_info)
            
            result.iloc[i] = signal_strength
    
    # Fill NaN values with 0
    result = result.fillna(0)
    
    return result
