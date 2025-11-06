import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Frequency Fractal Momentum with Liquidity Microstructure alpha factor
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Fractal Price Structure Analysis
    # Multi-timeframe fractal dimension estimation
    def hurst_exponent_hl(series, window):
        """Calculate Hurst exponent using high-low range scaling"""
        lags = range(2, min(window, len(series))//2)
        tau = []
        for lag in lags:
            if len(series) >= lag:
                tau.append(np.std(np.diff(series, lag)))
        if len(tau) < 2:
            return 0.5
        try:
            poly = np.polyfit(np.log(lags[:len(tau)]), np.log(tau), 1)
            return poly[0]
        except:
            return 0.5
    
    def hurst_exponent_close(series, window):
        """Calculate Hurst exponent using close-to-close scaling"""
        returns = series.pct_change().dropna()
        if len(returns) < window:
            return 0.5
        try:
            lags = range(2, min(window, len(returns))//2)
            tau = [np.std(returns.diff(lag).dropna()) for lag in lags if len(returns) >= lag]
            if len(tau) < 2:
                return 0.5
            poly = np.polyfit(np.log(lags[:len(tau)]), np.log(tau), 1)
            return poly[0]
        except:
            return 0.5
    
    # Calculate Hurst exponents
    data['hurst_5d_hl'] = data['high'].rolling(window=5, min_periods=3).apply(
        lambda x: hurst_exponent_hl(x, 5), raw=False
    )
    data['hurst_10d_close'] = data['close'].rolling(window=10, min_periods=5).apply(
        lambda x: hurst_exponent_close(x, 10), raw=False
    )
    data['fractal_consistency'] = data['hurst_5d_hl'] / data['hurst_10d_close']
    data['fractal_consistency'] = data['fractal_consistency'].replace([np.inf, -np.inf], 1).fillna(1)
    
    # Price path complexity measurement
    def zigzag_count(prices):
        """Count local extrema in price series"""
        if len(prices) < 3:
            return 0
        count = 0
        for i in range(1, len(prices)-1):
            if (prices[i] > prices[i-1] and prices[i] > prices[i+1]) or \
               (prices[i] < prices[i-1] and prices[i] < prices[i+1]):
                count += 1
        return count
    
    # Calculate intraday price changes using OHLC
    data['intraday_range'] = data['high'] - data['low']
    data['net_change'] = data['close'] - data['open']
    data['path_efficiency'] = data['net_change'] / (data['intraday_range'] + 1e-8)
    data['path_efficiency'] = data['path_efficiency'].replace([np.inf, -np.inf], 0).fillna(0)
    
    data['fractal_momentum'] = data['hurst_5d_hl'] * data['path_efficiency']
    
    # Multi-scale trend alignment
    data['micro_trend'] = np.sign(data['close'].pct_change(3))
    data['meso_trend'] = np.sign(data['close'].pct_change(8))
    data['macro_trend'] = np.sign(data['close'].pct_change(21))
    
    # Liquidity Microstructure Dynamics
    # Order flow imbalance estimation
    data['vw_price_pressure'] = (data['volume'] * (data['close'] - data['open'])) / (data['volume'] + 1e-8)
    data['spread_proxy'] = 2 * np.abs((data['high'] + data['low'])/2 - data['close']) / ((data['high'] + data['low'])/2 + 1e-8)
    data['micro_price_momentum'] = data['vw_price_pressure'] * (1 - data['spread_proxy'])
    
    # Market depth resilience
    avg_trade_size = data['volume'].rolling(window=5, min_periods=3).mean()
    data['absorption_capacity'] = data['intraday_range'] / (avg_trade_size + 1e-8)
    
    # Volume clustering persistence (autocorrelation at lag 1)
    data['volume_autocorr'] = data['volume'].rolling(window=10, min_periods=5).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
    ).fillna(0)
    
    data['depth_quality'] = data['absorption_capacity'] * data['volume_autocorr']
    
    # Liquidity regime classification
    depth_quality_median = data['depth_quality'].rolling(window=20, min_periods=10).median()
    data['liquidity_regime'] = (data['depth_quality'] > depth_quality_median).astype(int)
    data['liquidity_momentum'] = data['depth_quality'].diff(5)
    
    # Multi-frequency Momentum Convergence
    # Fractal-momentum alignment scoring
    data['micro_meso_alignment'] = data['micro_trend'] * data['meso_trend']
    data['meso_macro_alignment'] = data['meso_trend'] * data['macro_trend']
    data['full_alignment'] = data['micro_meso_alignment'] * data['meso_macro_alignment']
    
    # Momentum quality assessment
    daily_vol = data['close'].pct_change().rolling(window=5, min_periods=3).std()
    five_day_vol = data['close'].pct_change(5).rolling(window=5, min_periods=3).std()
    data['smoothness'] = 1 - (daily_vol / (five_day_vol + 1e-8))
    data['smoothness'] = data['smoothness'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Persistence strength (consecutive same-sign returns)
    def persistence_count(returns):
        if len(returns) < 2:
            return 1
        count = 1
        for i in range(1, len(returns)):
            if np.sign(returns.iloc[i]) == np.sign(returns.iloc[i-1]):
                count += 1
            else:
                break
        return count
    
    data['persistence'] = data['close'].pct_change().rolling(window=5, min_periods=3).apply(
        persistence_count, raw=False
    ).fillna(1)
    
    data['quality_momentum'] = data['fractal_momentum'] * data['smoothness'] * data['persistence']
    
    # Convergence acceleration
    data['fractal_acceleration'] = data['hurst_5d_hl'].diff(3)
    data['momentum_acceleration'] = data['quality_momentum'].diff(3)
    data['convergence_strength'] = data['fractal_acceleration'] * data['momentum_acceleration']
    
    # Liquidity-Adaptive Signal Construction
    # Regime-dependent weighting
    high_liquidity_weight_fractal = 0.6
    high_liquidity_weight_conv = 0.4
    low_liquidity_weight_fractal = 0.3
    low_liquidity_weight_conv = 0.7
    
    # Transition weighting based on liquidity momentum
    liquidity_momentum_norm = (data['liquidity_momentum'] - data['liquidity_momentum'].rolling(window=20).min()) / \
                             (data['liquidity_momentum'].rolling(window=20).max() - data['liquidity_momentum'].rolling(window=20).min() + 1e-8)
    liquidity_momentum_norm = liquidity_momentum_norm.fillna(0.5).clip(0, 1)
    
    data['weight_fractal'] = np.where(
        data['liquidity_regime'] == 1,
        high_liquidity_weight_fractal,
        low_liquidity_weight_fractal
    ) + (liquidity_momentum_norm - 0.5) * 0.2
    
    data['weight_conv'] = 1 - data['weight_fractal']
    
    # Core signal composition
    data['base_momentum'] = (data['weight_fractal'] * data['fractal_momentum'] + 
                           data['weight_conv'] * data['convergence_strength'])
    data['core_signal'] = data['base_momentum'] * data['micro_price_momentum']
    
    # Signal quality enhancement
    data['signal_aligned'] = data['core_signal'] * data['full_alignment']
    data['signal_depth_adjusted'] = data['signal_aligned'] / (data['absorption_capacity'] + 1e-8)
    data['signal_volume_validated'] = data['signal_depth_adjusted'] * data['volume_autocorr']
    
    # Multi-scale Pattern Integration
    # Timeframe synchronization analysis (simplified)
    micro_returns = data['close'].pct_change(3)
    meso_returns = data['close'].pct_change(8)
    
    # Calculate correlation between timeframes
    data['wavelet_coherence'] = micro_returns.rolling(window=10, min_periods=5).corr(meso_returns)
    data['wavelet_coherence'] = data['wavelet_coherence'].fillna(0)
    
    # Phase alignment (simplified as correlation sign)
    data['phase_alignment'] = np.sign(data['wavelet_coherence'])
    data['synchronization_strength'] = data['wavelet_coherence'] * data['phase_alignment']
    
    # Pattern persistence scoring
    data['fractal_stability'] = data['hurst_5d_hl'].rolling(window=5, min_periods=3).var().fillna(0)
    data['momentum_consistency'] = data['quality_momentum'].rolling(window=5, min_periods=3).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
    ).fillna(0)
    
    data['pattern_quality'] = (1 - data['fractal_stability']) * data['momentum_consistency']
    
    # Multi-scale convergence indicator
    data['scale_alignment_momentum'] = data['synchronization_strength'] * data['pattern_quality']
    data['fractal_efficiency'] = data['path_efficiency'] * data['fractal_consistency']
    data['integrated_convergence'] = data['scale_alignment_momentum'] * data['fractal_efficiency']
    
    # Final Alpha Generation
    data['primary_signal'] = data['signal_volume_validated'] * data['integrated_convergence']
    data['liquidity_filtered'] = data['primary_signal'] * data['depth_quality']
    data['pattern_validated'] = data['liquidity_filtered'] * data['full_alignment']
    data['microstructure_adjusted'] = data['pattern_validated'] / (data['spread_proxy'] + 1e-8)
    
    # Final alpha factor
    alpha = data['microstructure_adjusted'].replace([np.inf, -np.inf], 0).fillna(0)
    
    return alpha
