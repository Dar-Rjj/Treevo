import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Price-Volume Fractal Momentum Divergence factor
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Helper function for fractal dimension estimation using Hurst exponent approximation
    def estimate_fractal_dimension(series, window):
        """Estimate fractal dimension using rescaled range analysis"""
        if len(series) < window:
            return pd.Series([np.nan] * len(series), index=series.index)
        
        results = []
        for i in range(len(series)):
            if i < window - 1:
                results.append(np.nan)
                continue
                
            window_data = series.iloc[i-window+1:i+1]
            if len(window_data) < 2:
                results.append(np.nan)
                continue
                
            # Calculate rescaled range
            mean_val = window_data.mean()
            deviations = window_data - mean_val
            cumulative_deviations = deviations.cumsum()
            range_val = cumulative_deviations.max() - cumulative_deviations.min()
            std_val = window_data.std()
            
            if std_val == 0 or range_val == 0:
                hurst = 0.5
            else:
                hurst = np.log(range_val / std_val) / np.log(window)
            
            # Convert Hurst to fractal dimension: D = 2 - H
            fractal_dim = 2 - hurst
            results.append(fractal_dim)
        
        return pd.Series(results, index=series.index)
    
    # Calculate price fractal dimensions at different timeframes
    close_prices = data['close']
    
    # Short-term price fractal (2-5 days)
    price_fractal_short = estimate_fractal_dimension(close_prices, 5)
    
    # Medium-term price fractal (8-15 days)
    price_fractal_medium = estimate_fractal_dimension(close_prices, 15)
    
    # Long-term price fractal (25-50 days)
    price_fractal_long = estimate_fractal_dimension(close_prices, 35)
    
    # Calculate volume fractal dimensions
    volume = data['volume']
    
    # Short-term volume fractal (1-4 days)
    volume_fractal_short = estimate_fractal_dimension(volume, 4)
    
    # Medium-term volume fractal (6-12 days)
    volume_fractal_medium = estimate_fractal_dimension(volume, 12)
    
    # Long-term volume fractal (18-35 days)
    volume_fractal_long = estimate_fractal_dimension(volume, 25)
    
    # Calculate volume concentration patterns
    def volume_concentration(volume_series, window):
        """Measure volume concentration using Gini coefficient approximation"""
        concentration = pd.Series(index=volume_series.index, dtype=float)
        for i in range(len(volume_series)):
            if i < window - 1:
                concentration.iloc[i] = np.nan
                continue
                
            window_vol = volume_series.iloc[i-window+1:i+1]
            if window_vol.sum() == 0:
                concentration.iloc[i] = 0
            else:
                # Simplified Gini coefficient calculation
                sorted_vol = np.sort(window_vol)
                n = len(sorted_vol)
                concentration.iloc[i] = (2 * np.sum((np.arange(1, n+1) * sorted_vol)) / 
                                       (n * np.sum(sorted_vol))) - (n + 1) / n
        
        return concentration
    
    # Volume concentration at different timeframes
    vol_conc_short = volume_concentration(volume, 4)
    vol_conc_medium = volume_concentration(volume, 12)
    vol_conc_long = volume_concentration(volume, 25)
    
    # Calculate price-volume fractal divergence
    def fractal_divergence(price_fractal, volume_fractal, window=5):
        """Calculate divergence between price and volume fractal dimensions"""
        divergence = pd.Series(index=price_fractal.index, dtype=float)
        for i in range(len(price_fractal)):
            if i < window - 1:
                divergence.iloc[i] = np.nan
                continue
                
            # Correlation of recent changes
            price_changes = price_fractal.iloc[i-window+1:i+1].diff().dropna()
            volume_changes = volume_fractal.iloc[i-window+1:i+1].diff().dropna()
            
            if len(price_changes) < 2 or len(volume_changes) < 2:
                divergence.iloc[i] = 0
            else:
                # Negative correlation indicates divergence
                corr = price_changes.corr(volume_changes)
                if pd.isna(corr):
                    divergence.iloc[i] = 0
                else:
                    divergence.iloc[i] = -corr  # Negative correlation = divergence
        
        return divergence
    
    # Multi-timeframe divergences
    divergence_short = fractal_divergence(price_fractal_short, volume_fractal_short, 5)
    divergence_medium = fractal_divergence(price_fractal_medium, volume_fractal_medium, 8)
    divergence_long = fractal_divergence(price_fractal_long, volume_fractal_long, 12)
    
    # Calculate fractal momentum
    def fractal_momentum(fractal_series, short_window=3, long_window=8):
        """Calculate momentum in fractal dimension"""
        short_ma = fractal_series.rolling(window=short_window, min_periods=1).mean()
        long_ma = fractal_series.rolling(window=long_window, min_periods=1).mean()
        return (short_ma - long_ma) / long_ma
    
    price_momentum_short = fractal_momentum(price_fractal_short)
    price_momentum_medium = fractal_momentum(price_fractal_medium)
    price_momentum_long = fractal_momentum(price_fractal_long)
    
    volume_momentum_short = fractal_momentum(volume_fractal_short)
    volume_momentum_medium = fractal_momentum(volume_fractal_medium)
    volume_momentum_long = fractal_momentum(volume_fractal_long)
    
    # Calculate multi-scale fractal coherence
    def fractal_coherence(fractal_series_list, window=10):
        """Measure coherence across different fractal timeframes"""
        coherence = pd.Series(index=fractal_series_list[0].index, dtype=float)
        for i in range(len(coherence)):
            if i < window - 1:
                coherence.iloc[i] = np.nan
                continue
                
            # Calculate pairwise correlations between fractal series
            corrs = []
            for j in range(len(fractal_series_list)):
                for k in range(j+1, len(fractal_series_list)):
                    series1 = fractal_series_list[j].iloc[i-window+1:i+1]
                    series2 = fractal_series_list[k].iloc[i-window+1:i+1]
                    corr = series1.corr(series2)
                    if not pd.isna(corr):
                        corrs.append(corr)
            
            if len(corrs) > 0:
                coherence.iloc[i] = np.mean(corrs)
            else:
                coherence.iloc[i] = 0
        
        return coherence
    
    price_coherence = fractal_coherence([price_fractal_short, price_fractal_medium, price_fractal_long])
    volume_coherence = fractal_coherence([volume_fractal_short, volume_fractal_medium, volume_fractal_long])
    
    # Generate composite factor
    # 1. Raw fractal momentum divergence
    raw_divergence = (divergence_short * 0.4 + divergence_medium * 0.35 + divergence_long * 0.25)
    
    # 2. Multi-scale fractal consistency
    fractal_consistency = (price_coherence * 0.6 + volume_coherence * 0.4)
    
    # 3. Volume distribution confirmation
    volume_confirmation = (vol_conc_short * 0.3 + vol_conc_medium * 0.4 + vol_conc_long * 0.3)
    
    # 4. Fractal regime signals
    regime_signal = (
        (price_momentum_short * volume_momentum_short * 0.3) +
        (price_momentum_medium * volume_momentum_medium * 0.4) +
        (price_momentum_long * volume_momentum_long * 0.3)
    )
    
    # Final composite factor
    factor = (
        raw_divergence * 0.35 +
        fractal_consistency * 0.25 +
        volume_confirmation * 0.20 +
        regime_signal * 0.20
    )
    
    # Normalize the factor using rolling z-score (21-day window)
    factor_normalized = (factor - factor.rolling(window=21, min_periods=1).mean()) / factor.rolling(window=21, min_periods=1).std()
    
    return factor_normalized
