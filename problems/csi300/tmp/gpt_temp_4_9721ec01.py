import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    """
    Fractal Momentum Alpha Factor combining volatility regimes, price/volume fractal analysis,
    asymmetric momentum, and regime-weighted signals.
    """
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(60, len(df)):
        current_data = df.iloc[:i+1].copy()
        
        # 1. Volatility Fractal Regime
        returns = current_data['close'].pct_change()
        
        # Volatility ratios
        vol_5d = returns.iloc[-5:].std()
        vol_20d = returns.iloc[-20:].std()
        vol_60d = returns.iloc[-60:].std()
        
        vol_ratio_5_20 = vol_5d / vol_20d if vol_20d > 0 else 1.0
        vol_ratio_20_60 = vol_20d / vol_60d if vol_60d > 0 else 1.0
        
        # 2. Price Fractal Dimension
        # Hurst Exponent (20-day window)
        price_window = current_data['close'].iloc[-20:]
        if len(price_window) >= 20:
            lags = range(2, 10)
            tau = []
            for lag in lags:
                returns_lag = price_window.pct_change(lag).dropna()
                if len(returns_lag) > 0:
                    tau.append(np.std(returns_lag))
            
            if len(tau) > 1:
                H, _, _, _, _ = linregress(np.log(lags[:len(tau)]), np.log(tau))
                hurst_price = max(0.1, min(0.9, H))
            else:
                hurst_price = 0.5
        else:
            hurst_price = 0.5
        
        # Price Path Complexity (approximated by normalized range)
        price_range = (price_window.max() - price_window.min()) / price_window.mean() if price_window.mean() > 0 else 0
        price_complexity = min(1.0, price_range * 10)
        
        # 3. Asymmetric Momentum
        recent_returns = returns.iloc[-5:]
        upside_momentum = recent_returns[recent_returns > 0].mean() if len(recent_returns[recent_returns > 0]) > 0 else 0
        downside_momentum = recent_returns[recent_returns < 0].mean() if len(recent_returns[recent_returns < 0]) > 0 else 0
        
        momentum_asymmetry = (upside_momentum - abs(downside_momentum)) / (abs(upside_momentum) + abs(downside_momentum) + 1e-8)
        
        # 4. Volume Fractal Analysis
        volume_window = current_data['volume'].iloc[-20:]
        if len(volume_window) >= 20:
            # Volume Hurst Exponent approximation
            volume_changes = volume_window.pct_change().dropna()
            if len(volume_changes) >= 10:
                volume_std_short = volume_changes.iloc[-5:].std()
                volume_std_long = volume_changes.std()
                volume_hurst = volume_std_short / (volume_std_long + 1e-8)
                volume_hurst = max(0.1, min(0.9, volume_hurst))
            else:
                volume_hurst = 0.5
        else:
            volume_hurst = 0.5
        
        # Volume Clustering (autocorrelation of volume changes)
        if len(volume_changes) >= 5:
            volume_clustering = volume_changes.autocorr(lag=1) if not pd.isna(volume_changes.autocorr(lag=1)) else 0
        else:
            volume_clustering = 0
        
        # 5. Price-Volume Fractal Sync
        fractal_dim_diff = abs(hurst_price - volume_hurst)
        
        # Synchronization Score (correlation between price and volume movements)
        if len(returns) >= 20 and len(volume_changes) >= 20:
            price_volume_corr = returns.iloc[-20:].corr(volume_changes.iloc[-20:])
            sync_score = 0 if pd.isna(price_volume_corr) else price_volume_corr
        else:
            sync_score = 0
        
        # 6. Regime-Weighted Signal
        # Regime classification based on volatility ratios
        high_vol_regime = vol_ratio_5_20 > 1.2 or vol_ratio_20_60 > 1.1
        low_vol_regime = vol_ratio_5_20 < 0.8 and vol_ratio_20_60 < 0.9
        
        # Base momentum signal
        base_signal = momentum_asymmetry * (1 + price_complexity)
        
        # Regime adjustments
        if high_vol_regime:
            # In high volatility, emphasize mean reversion and volume signals
            regime_weight = 0.3 * base_signal + 0.4 * volume_clustering + 0.3 * sync_score
        elif low_vol_regime:
            # In low volatility, emphasize trend following and fractal structure
            regime_weight = 0.5 * base_signal + 0.3 * hurst_price + 0.2 * (1 - fractal_dim_diff)
        else:
            # Normal regime - balanced approach
            regime_weight = 0.4 * base_signal + 0.2 * hurst_price + 0.2 * volume_hurst + 0.2 * sync_score
        
        # Final alpha score with volatility normalization
        final_alpha = regime_weight / (vol_20d + 1e-8)
        
        result.iloc[i] = final_alpha
    
    # Forward fill any remaining NaN values
    result = result.ffill().fillna(0)
    
    return result
