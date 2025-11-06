import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    """
    Multi-Scale Price-Volume Divergence with Liquidity Acceleration factor
    """
    data = df.copy()
    
    # Fractal Price Efficiency
    def hurst_exponent(series, window):
        """Calculate Hurst exponent for price series"""
        lags = range(2, min(15, window//2))
        tau = []
        for lag in lags:
            # RS analysis
            series_lag = series.diff(lag).dropna()
            if len(series_lag) < 2:
                return 0.5
            R = series_lag.max() - series_lag.min()
            S = series_lag.std()
            if S == 0:
                tau.append(0)
            else:
                tau.append(np.log(R/S))
        
        if len(tau) < 2:
            return 0.5
            
        # Linear regression for Hurst exponent
        try:
            slope, _, _, _, _ = linregress(np.log(lags[:len(tau)]), tau)
            return slope
        except:
            return 0.5
    
    # Multi-scale Hurst exponents
    hurst_5 = data['close'].rolling(window=5).apply(lambda x: hurst_exponent(x, 5), raw=False)
    hurst_10 = data['close'].rolling(window=10).apply(lambda x: hurst_exponent(x, 10), raw=False)
    hurst_20 = data['close'].rolling(window=20).apply(lambda x: hurst_exponent(x, 20), raw=False)
    
    # Fractal dimension approximation
    fractal_dim = 2 - ((hurst_5 + hurst_10 + hurst_20) / 3)
    price_efficiency = 1.0 / (1.0 + np.abs(fractal_dim - 1.5))
    
    # Volume-Price Divergence
    # Directional volume imbalance
    price_ret = data['close'].pct_change()
    up_days = price_ret > 0
    down_days = price_ret < 0
    
    up_volume = data['volume'].where(up_days, 0).rolling(window=10).mean()
    down_volume = data['volume'].where(down_days, 0).rolling(window=10).mean()
    volume_imbalance = (up_volume - down_volume) / (up_volume + down_volume + 1e-8)
    
    # Volume concentration on price moves
    large_moves = np.abs(price_ret) > price_ret.rolling(window=20).std()
    concentrated_volume = data['volume'].where(large_moves, 0).rolling(window=10).sum() / \
                         data['volume'].rolling(window=10).sum()
    
    # Volume leadership indicator
    volume_change = data['volume'].pct_change()
    price_change = price_ret.shift(-1)  # Future price change for correlation (allowed as it's past data)
    
    # Cross-correlation at different lags
    corr_lag1 = data['volume'].rolling(window=10).corr(price_ret.shift(1))
    corr_lag0 = data['volume'].rolling(window=10).corr(price_ret)
    
    volume_leadership = (corr_lag1 - corr_lag0).fillna(0)
    
    # Liquidity Acceleration
    # Order flow intensity - Amount-per-trade momentum
    amount_per_trade = data['amount'] / (data['volume'] + 1e-8)
    trade_size_momentum = amount_per_trade.pct_change(3).rolling(window=5).mean()
    
    # Trade size distribution skewness approximation
    trade_size_skew = (amount_per_trade - amount_per_trade.rolling(window=10).mean()) / \
                     (amount_per_trade.rolling(window=10).std() + 1e-8)
    trade_size_skew = trade_size_skew.rolling(window=5).apply(
        lambda x: x.skew() if len(x) > 2 else 0, raw=False
    )
    
    # Liquidity momentum via spread compression proxy
    high_low_range = (data['high'] - data['low']) / data['close']
    range_momentum = -high_low_range.pct_change(3)  # Negative because compression is good
    
    # Market depth changes proxy using volume concentration
    volume_concentration = data['volume'].rolling(window=5).std() / \
                          (data['volume'].rolling(window=5).mean() + 1e-8)
    depth_momentum = -volume_concentration.pct_change(2)
    
    # Microstructure Regime Identification
    # Tick-level activity clustering proxy
    volume_volatility = data['volume'].pct_change().rolling(window=5).std()
    price_volatility = price_ret.rolling(window=5).std()
    
    # High-frequency regime detection
    hf_regime = ((volume_volatility > volume_volatility.rolling(window=20).quantile(0.7)) & 
                (price_volatility > price_volatility.rolling(window=20).quantile(0.6))).astype(float)
    
    # Trade interval analysis proxy
    volume_gaps = (data['volume'].shift(1) / (data['volume'] + 1e-8))
    activity_clustering = volume_gaps.rolling(window=5).std()
    
    # Institutional flow detection proxy
    large_trade_ratio = (data['amount'].where(amount_per_trade > amount_per_trade.rolling(window=20).quantile(0.8), 0) 
                        / (data['amount'] + 1e-8)).rolling(window=5).mean()
    
    # Dynamic Signal Integration
    # Combine fractal efficiency × volume leadership
    base_signal = price_efficiency * volume_leadership
    
    # Liquidity regime adjustments
    liquidity_acceleration = (trade_size_momentum + range_momentum + depth_momentum) / 3
    
    # Microstructure regime timing
    regime_multiplier = 1.0 + (hf_regime * 0.3)  # Amplify during high-frequency regimes
    
    # Final factor integration
    factor = (base_signal * liquidity_acceleration * regime_multiplier * 
             (1.0 + 0.2 * volume_imbalance) * (1.0 + 0.1 * concentrated_volume))
    
    # Normalize and clean
    factor = factor.replace([np.inf, -np.inf], np.nan)
    factor = (factor - factor.rolling(window=20).mean()) / (factor.rolling(window=20).std() + 1e-8)
    
    return factor
