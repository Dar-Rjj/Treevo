import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Fractal Efficiency with Volume-Price Dynamics alpha factor
    """
    # Make a copy to avoid modifying original dataframe
    data = df.copy()
    
    # Multi-Scale Fractal Analysis
    def hurst_exponent(series, window):
        """Calculate Hurst exponent using R/S analysis"""
        lags = range(2, min(window, 20))
        tau = []
        for lag in lags:
            # Calculate lagged differences
            ds = series.diff(lag).dropna()
            if len(ds) < 2:
                tau.append(np.nan)
                continue
            # R/S analysis
            mean_ds = ds.mean()
            deviations = ds - mean_ds
            Z = deviations.cumsum()
            R = Z.max() - Z.min()
            S = ds.std()
            if S > 0:
                tau.append(np.log(R/S))
            else:
                tau.append(np.nan)
        
        if len([x for x in tau if not np.isnan(x)]) < 2:
            return np.nan
        
        # Linear regression to get Hurst exponent
        valid_lags = [l for l, t in zip(lags, tau) if not np.isnan(t)]
        valid_tau = [t for t in tau if not np.isnan(t)]
        
        if len(valid_tau) < 2:
            return np.nan
            
        hurst = np.polyfit(np.log(valid_lags), valid_tau, 1)[0]
        return hurst
    
    # Calculate fractal dimensions across multiple windows
    for window in [5, 10, 15]:
        data[f'fractal_dim_{window}'] = data['close'].rolling(window=window).apply(
            lambda x: 2 - hurst_exponent(x, window) if len(x) == window else np.nan, 
            raw=False
        )
    
    # Volume fractal properties
    def volume_fractal(volume_series, window):
        """Calculate volume clustering fractal dimension"""
        if len(volume_series) < window:
            return np.nan
        
        # Normalize volume
        vol_norm = volume_series / volume_series.mean()
        
        # Calculate burst intensity (volume above threshold)
        threshold = vol_norm.quantile(0.7)
        bursts = (vol_norm > threshold).astype(int)
        
        # Simple fractal measure using run lengths
        if bursts.sum() == 0:
            return 0
        
        # Count burst clusters
        changes = bursts.diff().fillna(0)
        cluster_starts = (changes == 1).sum()
        
        if cluster_starts == 0:
            return 0
            
        avg_cluster_size = bursts.sum() / cluster_starts
        fractal_dim = 1 - (avg_cluster_size / window)  # Higher = more fragmented
        
        return fractal_dim
    
    for window in [5, 10]:
        data[f'volume_fractal_{window}'] = data['volume'].rolling(window=window).apply(
            lambda x: volume_fractal(x, window), raw=False
        )
    
    # Regime Detection
    def detect_price_regime(close_prices, window=10):
        """Detect price regime using directional persistence"""
        if len(close_prices) < window:
            return 0
        
        returns = close_prices.pct_change().dropna()
        if len(returns) < 2:
            return 0
            
        # Directional persistence (autocorrelation of signed returns)
        signed_returns = np.sign(returns)
        persistence = signed_returns.rolling(window=min(5, len(signed_returns))).apply(
            lambda x: x.autocorr() if len(x) > 1 else 0, raw=False
        ).iloc[-1] if len(signed_returns) > 1 else 0
        
        # Cumulative return pattern
        cum_returns = (1 + returns).cumprod()
        trend_strength = (cum_returns.iloc[-1] / cum_returns.iloc[0] - 1) if len(cum_returns) > 0 else 0
        
        # Regime classification
        if abs(persistence) > 0.3 and abs(trend_strength) > 0.02:
            return 1  # Trending regime
        elif abs(persistence) < 0.1 and abs(trend_strength) < 0.01:
            return -1  # Mean-reverting regime
        else:
            return 0  # Transition regime
    
    def detect_volume_regime(volume_series, window=10):
        """Detect volume regime using momentum and acceleration"""
        if len(volume_series) < window:
            return 0
        
        # Volume momentum (5-day vs 10-day)
        vol_5d = volume_series.rolling(5).mean()
        vol_10d = volume_series.rolling(10).mean()
        vol_momentum = (vol_5d / vol_10d - 1).iloc[-1] if len(vol_5d) > 0 and len(vol_10d) > 0 else 0
        
        # Volume acceleration (rate of change of momentum)
        vol_accel = vol_momentum - (vol_5d.shift(1) / vol_10d.shift(1) - 1).iloc[-1] if len(vol_5d) > 1 else 0
        
        if abs(vol_momentum) > 0.2 and abs(vol_accel) > 0.05:
            return 1  # High activity regime
        elif abs(vol_momentum) < 0.05 and abs(vol_accel) < 0.01:
            return -1  # Low activity regime
        else:
            return 0  # Normal regime
    
    # Calculate regimes
    data['price_regime'] = data['close'].rolling(window=10).apply(
        lambda x: detect_price_regime(x), raw=False
    )
    
    data['volume_regime'] = data['volume'].rolling(window=10).apply(
        lambda x: detect_volume_regime(x), raw=False
    )
    
    # Unified regime framework
    data['unified_regime'] = np.where(
        (data['price_regime'] == 1) & (data['volume_regime'].isin([0, 1])), 1,  # Trending
        np.where(
            (data['price_regime'] == -1) & (data['volume_regime'].isin([0, -1])), -1,  # Mean-reverting
            0  # Transition
        )
    )
    
    # Regime-Specific Momentum Calculation
    def regime_momentum(close_prices, regime, windows=[3, 5, 10]):
        """Calculate regime-appropriate momentum"""
        if len(close_prices) < max(windows):
            return 0
        
        momentum_components = []
        
        for window in windows:
            if len(close_prices) >= window:
                mom = (close_prices.iloc[-1] / close_prices.iloc[-window] - 1)
                momentum_components.append(mom)
        
        if not momentum_components:
            return 0
            
        # Regime-specific weighting
        if regime == 1:  # Trending regime - emphasize longer-term momentum
            weights = [0.2, 0.3, 0.5] if len(momentum_components) == 3 else [1/len(momentum_components)] * len(momentum_components)
        elif regime == -1:  # Mean-reverting regime - emphasize shorter-term momentum
            weights = [0.5, 0.3, 0.2] if len(momentum_components) == 3 else [1/len(momentum_components)] * len(momentum_components)
        else:  # Transition regime - balanced weighting
            weights = [0.33, 0.33, 0.34] if len(momentum_components) == 3 else [1/len(momentum_components)] * len(momentum_components)
        
        weighted_momentum = sum(m * w for m, w in zip(momentum_components, weights))
        return weighted_momentum
    
    # Calculate regime-adaptive momentum
    data['regime_momentum'] = data.apply(
        lambda row: regime_momentum(
            data['close'].loc[:row.name].tail(10), 
            row['unified_regime'] if not pd.isna(row['unified_regime']) else 0
        ), axis=1
    )
    
    # Volume-Price Confirmation
    # Intraday range efficiency
    data['range_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['range_efficiency_5d'] = data['range_efficiency'].rolling(5).mean()
    
    # Volume-weighted breakout quality
    data['volume_ratio'] = data['volume'] / data['volume'].rolling(10).mean()
    data['breakout_quality'] = np.where(
        (data['close'] > data['high'].shift(1)) & (data['volume_ratio'] > 1.2),
        data['volume_ratio'] * (data['close'] / data['high'].shift(1) - 1),
        0
    )
    
    # Multi-Frequency Signal Integration
    # Cross-timeframe momentum convergence
    mom_3d = data['close'].pct_change(3)
    mom_5d = data['close'].pct_change(5)
    mom_10d = data['close'].pct_change(10)
    
    data['momentum_convergence'] = (
        np.sign(mom_3d) * np.sign(mom_5d) * np.sign(mom_10d) * 
        (abs(mom_3d) + abs(mom_5d) + abs(mom_10d)) / 3
    ).fillna(0)
    
    # Fractal-Momentum Synthesis
    fractal_dim_avg = data[['fractal_dim_5', 'fractal_dim_10', 'fractal_dim_15']].mean(axis=1)
    data['fractal_weighted_momentum'] = data['regime_momentum'] * fractal_dim_avg
    
    # Dynamic Alpha Factor Generation
    def generate_composite_alpha(row):
        """Generate final composite alpha factor"""
        if any(pd.isna([row.get('unified_regime', 0), row.get('regime_momentum', 0), 
                       row.get('fractal_weighted_momentum', 0), row.get('momentum_convergence', 0)])):
            return 0
        
        regime = row.get('unified_regime', 0)
        base_momentum = row.get('regime_momentum', 0)
        fractal_momentum = row.get('fractal_weighted_momentum', 0)
        momentum_conv = row.get('momentum_convergence', 0)
        range_eff = row.get('range_efficiency_5d', 0) or 0
        breakout_qual = row.get('breakout_quality', 0) or 0
        vol_fractal = row.get('volume_fractal_10', 0) or 0
        
        # Volume-price quality adjustment
        volume_confirmation = 1 + 0.5 * vol_fractal  # Higher fractal = more confirmation
        price_quality = 1 + 0.3 * (abs(range_eff) - 0.5)  # Center around 0.5 efficiency
        
        # Regime-adaptive signal combination
        if regime == 1:  # Trending regime
            alpha = (
                0.6 * fractal_momentum + 
                0.3 * momentum_conv + 
                0.1 * breakout_qual
            )
        elif regime == -1:  # Mean-reverting regime
            alpha = (
                0.4 * base_momentum + 
                0.4 * (-fractal_momentum) +  # Inverse for mean reversion
                0.2 * range_eff
            )
        else:  # Transition regime
            alpha = (
                0.4 * base_momentum + 
                0.3 * momentum_conv + 
                0.3 * fractal_momentum
            )
        
        # Apply volume-price quality adjustments
        alpha *= volume_confirmation * price_quality
        
        # Regime stability weighting
        regime_stability = abs(regime)  # More stable regimes get full weight
        alpha *= (0.5 + 0.5 * regime_stability)
        
        return alpha
    
    # Generate final alpha factor
    alpha_series = data.apply(generate_composite_alpha, axis=1)
    
    # Clean and normalize
    alpha_series = alpha_series.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Remove any potential lookahead bias by ensuring no future information
    alpha_series = alpha_series.shift(1).fillna(0)  # Shift to avoid using current day's close
    
    return alpha_series
