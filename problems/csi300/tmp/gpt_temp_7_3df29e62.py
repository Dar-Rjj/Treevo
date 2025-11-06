import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Helper function for fractal dimension approximation
    def hurst_exponent(series, max_lag=20):
        lags = range(2, min(max_lag, len(series)))
        tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]
    
    # Calculate fractal momentum components
    def fractal_breakout_momentum(data):
        high_roll = data['high'].rolling(window=10, min_periods=10).max().shift(1)
        low_roll = data['low'].rolling(window=10, min_periods=10).min().shift(1)
        range_10 = high_roll - low_roll
        breakout = (data['close'] - high_roll) / range_10
        
        # Fractal momentum using Hurst exponent
        close_series = data['close'].rolling(window=30, min_periods=30).apply(
            lambda x: hurst_exponent(x) if len(x) >= 10 else np.nan, raw=False
        )
        return breakout * close_series
    
    def volume_adapted_fractal(data):
        close_diff = data['close'] - data['close'].shift(5)
        high_low_range = data['high'].rolling(window=6).max() - data['low'].rolling(window=6).min()
        
        # Volume self-similarity using fractal dimension
        volume_series = data['volume'].rolling(window=20, min_periods=20).apply(
            lambda x: hurst_exponent(x) if len(x) >= 10 else np.nan, raw=False
        )
        return (close_diff / high_low_range) * volume_series
    
    def regime_aware_mean_reversion(data):
        close_median = data['close'].rolling(window=20, min_periods=20).median()
        high_low_range = data['high'].rolling(window=20).max() - data['low'].rolling(window=20).min()
        
        # Fractal volatility (inverse)
        volatility_series = data['close'].pct_change().rolling(window=20, min_periods=20).std()
        fractal_vol = 1 / (volatility_series + 1e-8)
        
        return ((data['close'] - close_median) / high_low_range) * fractal_vol
    
    # Microstructure fractal dynamics
    def fractal_order_flow(data):
        intraday_position = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8) - 0.5
        
        # Intraday fractal dimension using high-low range
        def intraday_fractal(high_low_series):
            if len(high_low_series) < 10:
                return np.nan
            return hurst_exponent(high_low_series)
        
        high_low_range = data['high'] - data['low']
        intraday_fractal_dim = high_low_range.rolling(window=15, min_periods=15).apply(
            intraday_fractal, raw=False
        )
        
        return intraday_position * intraday_fractal_dim
    
    def volume_price_fractal_efficiency(data):
        price_range = np.abs(data['close'] - data['open'])
        volume_ratio = data['volume'] / data['volume'].rolling(window=20, min_periods=20).mean()
        
        # Fractal liquidity using volume patterns
        volume_fractal = data['volume'].rolling(window=20, min_periods=20).apply(
            lambda x: hurst_exponent(x) if len(x) >= 10 else np.nan, raw=False
        )
        
        return (price_range / (volume_ratio + 1e-8)) * volume_fractal
    
    def large_trade_fractal_impact(data):
        trade_size = data['amount'] / (data['volume'] * data['close'] + 1e-8)
        price_impact = (data['high'] - data['low']) / (data['volume'] + 1e-8)
        
        # Volume self-similarity
        volume_similarity = data['volume'].rolling(window=20, min_periods=20).apply(
            lambda x: hurst_exponent(x) if len(x) >= 10 else np.nan, raw=False
        )
        
        return trade_size * price_impact * volume_similarity
    
    # Nonlinear volatility regimes
    def fractal_volatility_break(data):
        current_range = data['high'] - data['low']
        median_range = (data['high'] - data['low']).rolling(window=20, min_periods=20).median()
        
        range_deviation = np.abs(current_range - median_range) / (median_range + 1e-8)
        
        # Regime change detection using variance ratio
        def regime_change(close_series):
            if len(close_series) < 10:
                return np.nan
            returns = close_series.pct_change().dropna()
            if len(returns) < 5:
                return np.nan
            var_short = returns.rolling(window=5).var().iloc[-1]
            var_long = returns.rolling(window=10).var().iloc[-1]
            return var_short / (var_long + 1e-8)
        
        regime_detection = data['close'].rolling(window=15, min_periods=15).apply(
            regime_change, raw=False
        )
        
        return range_deviation * regime_detection
    
    def nonlinear_regime_momentum(data):
        price_change = data['close'] - data['close'].shift(5)
        direction = np.sign(price_change)
        price_range = data['high'].rolling(window=6).max() - data['low'].rolling(window=6).min()
        
        # Price-volume scaling correlation
        def price_volume_scaling(close_vol_data):
            if len(close_vol_data) < 10:
                return np.nan
            close_changes = close_vol_data['close'].pct_change().dropna()
            volume_changes = close_vol_data['volume'].pct_change().dropna()
            if len(close_changes) < 5:
                return np.nan
            return np.corrcoef(close_changes.tail(5), volume_changes.tail(5))[0,1]
        
        scaling_data = data[['close', 'volume']].rolling(window=10, min_periods=10).apply(
            lambda x: price_volume_scaling(pd.DataFrame(x, columns=['close', 'volume'])) 
            if len(x) >= 10 else np.nan, raw=False
        )
        
        return direction * (price_change / (price_range + 1e-8)) * scaling_data
    
    def high_low_fractal_volatility(data):
        range_cubed = (data['high'] - data['low']) ** 3
        normalized_vol = range_cubed / (data['close'] * data['volume'] + 1e-8)
        
        # Fractal momentum component
        fractal_mom = data['close'].rolling(window=20, min_periods=20).apply(
            lambda x: hurst_exponent(x) if len(x) >= 10 else np.nan, raw=False
        )
        
        return normalized_vol * fractal_mom
    
    # Calculate all components
    fbm = fractal_breakout_momentum(data)
    vaf = volume_adapted_fractal(data)
    ramr = regime_aware_mean_reversion(data)
    fof = fractal_order_flow(data)
    vpfe = volume_price_fractal_efficiency(data)
    ltfi = large_trade_fractal_impact(data)
    fvb = fractal_volatility_break(data)
    nrm = nonlinear_regime_momentum(data)
    hlfv = high_low_fractal_volatility(data)
    
    # Adaptive composite signals
    core_fractal_momentum = fbm * vaf * fof
    nonlinear_microstructure = vpfe * ltfi * hlfv
    regime_adaptation_framework = fvb * ramr * nrm
    
    # Final alpha integration
    final_alpha = core_fractal_momentum * nonlinear_microstructure * regime_adaptation_framework
    
    # Clean and return
    final_alpha = final_alpha.replace([np.inf, -np.inf], np.nan)
    return final_alpha
