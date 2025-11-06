import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volume-Price Fractal Efficiency Alpha Factor
    Combines fractal analysis of price movements with microstructure order flow signals
    """
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Required columns
    cols_required = ['open', 'high', 'low', 'close', 'volume', 'amount']
    if not all(col in df.columns for col in cols_required):
        return result
    
    # Calculate basic price metrics
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['range'] = (df['high'] - df['low']) / df['close'].shift(1)
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    
    # Calculate volume-weighted price
    df['vwap'] = (df['amount'] / df['volume']).fillna(df['close'])
    
    # 1. Fractal Dimension Analysis - Hurst Exponent approximation
    window_hurst = 20
    
    def hurst_approximation(series, window):
        """Approximate Hurst exponent using price range scaling"""
        hurst_values = []
        for i in range(len(series)):
            if i < window:
                hurst_values.append(np.nan)
                continue
            
            window_data = series.iloc[i-window+1:i+1]
            if len(window_data) < window:
                hurst_values.append(np.nan)
                continue
            
            # Calculate rescaled range
            mean_val = window_data.mean()
            deviations = window_data - mean_val
            cumulative_dev = deviations.cumsum()
            range_val = cumulative_dev.max() - cumulative_dev.min()
            std_val = window_data.std()
            
            if std_val == 0:
                hurst_values.append(0.5)
            else:
                hurst = np.log(range_val / std_val) / np.log(window)
                hurst_values.append(hurst)
        
        return pd.Series(hurst_values, index=series.index)
    
    df['hurst_price'] = hurst_approximation(df['close'], window_hurst)
    
    # 2. Volume Distribution Fractality - Volume clustering
    window_vol = 10
    
    def volume_clustering(volume_series, window):
        """Measure volume autocorrelation and clustering intensity"""
        clustering = []
        for i in range(len(volume_series)):
            if i < window:
                clustering.append(np.nan)
                continue
            
            window_data = volume_series.iloc[i-window+1:i+1]
            if len(window_data) < window:
                clustering.append(np.nan)
                continue
            
            # Calculate autocorrelation at lag 1
            autocorr = window_data.autocorr(lag=1)
            if pd.isna(autocorr):
                clustering.append(0)
            else:
                # Variance ratio for persistence detection
                returns_vol = volume_series.iloc[i-window+1:i+1].pct_change().dropna()
                if len(returns_vol) > 1:
                    var_ratio = returns_vol.var() / (returns_vol.iloc[::2].var() * 2) if len(returns_vol) >= 4 else 1
                    clustering_intensity = autocorr * (1 if var_ratio > 1 else -1)
                else:
                    clustering_intensity = autocorr
                clustering.append(clustering_intensity)
        
        return pd.Series(clustering, index=volume_series.index)
    
    df['volume_clustering'] = volume_clustering(df['volume'], window_vol)
    
    # 3. Order Flow Asymmetry - Tick rule with volume confirmation
    def calculate_order_flow(df):
        """Detect buyer vs seller initiated volume using tick rule"""
        order_flow = []
        for i in range(len(df)):
            if i == 0:
                order_flow.append(0)
                continue
            
            current_close = df['close'].iloc[i]
            prev_close = df['close'].iloc[i-1]
            current_volume = df['volume'].iloc[i]
            
            # Tick rule: if price up, buyer initiated; if down, seller initiated
            if current_close > prev_close:
                flow = current_volume  # Buyer initiated
            elif current_close < prev_close:
                flow = -current_volume  # Seller initiated
            else:
                # No price change, use volume-weighted price movement
                vwap_change = df['vwap'].iloc[i] - df['vwap'].iloc[i-1]
                if vwap_change > 0:
                    flow = current_volume * 0.5
                elif vwap_change < 0:
                    flow = -current_volume * 0.5
                else:
                    flow = 0
            
            order_flow.append(flow)
        
        return pd.Series(order_flow, index=df.index)
    
    df['order_flow'] = calculate_order_flow(df)
    
    # Order flow momentum (5-day rolling sum)
    df['order_flow_momentum'] = df['order_flow'].rolling(window=5, min_periods=3).sum()
    
    # 4. Price Impact of Volume
    window_impact = 10
    
    def volume_price_impact(df, window):
        """Measure immediate price impact of volume"""
        impact_values = []
        for i in range(len(df)):
            if i < window:
                impact_values.append(np.nan)
                continue
            
            window_data = df.iloc[i-window+1:i+1]
            if len(window_data) < window:
                impact_values.append(np.nan)
                continue
            
            # Calculate correlation between volume changes and price changes
            vol_changes = window_data['volume'].pct_change().dropna()
            price_changes = window_data['returns'].dropna()
            
            if len(vol_changes) > 2 and len(price_changes) > 2:
                min_len = min(len(vol_changes), len(price_changes))
                correlation = np.corrcoef(vol_changes.iloc[-min_len:], 
                                        price_changes.iloc[-min_len:])[0,1]
                if pd.isna(correlation):
                    impact = 0
                else:
                    impact = correlation
            else:
                impact = 0
            
            impact_values.append(impact)
        
        return pd.Series(impact_values, index=df.index)
    
    df['volume_price_impact'] = volume_price_impact(df, window_impact)
    
    # 5. Fractal Regime Classification
    def classify_fractal_regime(hurst, volume_clustering):
        """Classify market regime based on fractal characteristics"""
        if pd.isna(hurst) or pd.isna(volume_clustering):
            return 0
        
        # Trending regime: high Hurst + positive volume clustering
        if hurst > 0.6 and volume_clustering > 0.1:
            return 1  # Strong trending
        elif hurst > 0.55 and volume_clustering > 0:
            return 0.5  # Moderate trending
        
        # Mean-reverting regime: low Hurst + negative volume clustering
        elif hurst < 0.45 and volume_clustering < -0.1:
            return -1  # Strong mean-reversion
        elif hurst < 0.5 and volume_clustering < 0:
            return -0.5  # Moderate mean-reversion
        
        # Congested/transition regime
        else:
            return 0
    
    df['fractal_regime'] = [classify_fractal_regime(h, vc) for h, vc in 
                           zip(df['hurst_price'], df['volume_clustering'])]
    
    # 6. Composite Alpha Factor Generation
    for i in range(len(df)):
        if (pd.isna(df['hurst_price'].iloc[i]) or 
            pd.isna(df['order_flow_momentum'].iloc[i]) or
            pd.isna(df['volume_price_impact'].iloc[i]) or
            pd.isna(df['fractal_regime'].iloc[i])):
            result.iloc[i] = np.nan
            continue
        
        # Base factor: fractal efficiency adjusted by order flow
        base_factor = df['hurst_price'].iloc[i] * df['order_flow_momentum'].iloc[i]
        
        # Apply regime-dependent weighting
        regime = df['fractal_regime'].iloc[i]
        
        if regime > 0:  # Trending regime
            # Emphasize directional persistence
            weight = 1.2 + (regime * 0.3)
            momentum_component = df['order_flow_momentum'].iloc[i] * weight
            
        elif regime < 0:  # Mean-reverting regime
            # Emphasize mean-reversion signals (inverse momentum)
            weight = 0.8 + (abs(regime) * 0.4)
            momentum_component = -df['order_flow_momentum'].iloc[i] * weight
            
        else:  # Congested/transition regime
            # Neutral weighting with volume impact adjustment
            momentum_component = df['order_flow_momentum'].iloc[i] * df['volume_price_impact'].iloc[i]
        
        # Final composite factor
        composite_factor = base_factor * 0.6 + momentum_component * 0.4
        
        result.iloc[i] = composite_factor
    
    # Normalize the final factor
    result = (result - result.rolling(window=20, min_periods=10).mean()) / result.rolling(window=20, min_periods=10).std()
    
    return result
