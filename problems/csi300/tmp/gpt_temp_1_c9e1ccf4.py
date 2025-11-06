import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Multi-Scale Volatility Dynamics
    data['micro_vol'] = (data['high'] - data['low']) / data['close'].shift(1)
    
    # Macro Volatility (10-day)
    returns = data['close'].pct_change()
    data['macro_vol'] = returns.rolling(window=10).apply(lambda x: np.sqrt(np.sum((x)**2)), raw=True)
    
    # Volatility Efficiency
    data['vol_efficiency'] = abs(data['close'] - data['close'].shift(1)) / (data['high'] - data['low'])
    data['vol_efficiency'] = data['vol_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    # Fractal Momentum Structure
    # Short-Term Hurst (5-day)
    def hurst_exponent(series):
        if len(series) < 5:
            return np.nan
        range_val = series.max() - series.min()
        std_val = series.std()
        if std_val == 0:
            return np.nan
        return np.log(range_val / std_val) / np.log(len(series))
    
    data['hurst_short'] = data['close'].rolling(window=5).apply(hurst_exponent, raw=True)
    
    # Medium-Term Hurst (20-day)
    data['hurst_medium'] = data['close'].rolling(window=20).apply(hurst_exponent, raw=True)
    
    # Fractal Dimension Change
    data['fractal_change'] = data['hurst_medium'] - data['hurst_short']
    
    # Gap Pressure Analysis
    data['gap_magnitude'] = abs(data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Gap Fill Ratio
    def calculate_gap_fill(row):
        if row.name == 0:
            return np.nan
        prev_close = data.loc[row.name - pd.Timedelta(days=1), 'close'] if isinstance(data.index, pd.DatetimeIndex) else data['close'].shift(1).loc[row.name]
        open_price = row['open']
        
        if open_price > prev_close:  # Up gap
            gap = open_price - prev_close
            if gap == 0:
                return np.nan
            fill_amount = open_price - row['low']
            return fill_amount / gap
        else:  # Down gap
            gap = prev_close - open_price
            if gap == 0:
                return np.nan
            fill_amount = row['high'] - open_price
            return fill_amount / gap
    
    data['gap_fill_ratio'] = data.apply(calculate_gap_fill, axis=1)
    
    # Pressure-Momentum Alignment
    data['vwap'] = data['amount'] / data['volume']
    data['vwap_change'] = data['vwap'] / data['vwap'].shift(1) - 1
    data['price_change'] = data['close'] / data['close'].shift(1) - 1
    data['pressure_momentum'] = np.sign(data['vwap_change']) * np.sign(data['price_change'])
    
    # Volume-Fractal Dynamics
    data['volume_momentum'] = data['volume'] / data['volume'].shift(5) - 1
    
    # Volume Efficiency
    data['volume_efficiency'] = data['volume'] / (data['high'] - data['low'])
    data['volume_efficiency'] = data['volume_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    # Fractal Volume Pattern
    data['fractal_volume'] = data['volume'] / data['volume'].rolling(window=5).mean()
    
    # Regime Detection
    vol_median = data['macro_vol'].median()
    data['high_vol_regime'] = data['macro_vol'] > vol_median * 1.2
    data['low_vol_regime'] = data['macro_vol'] < vol_median * 0.8
    
    # High Volatility Regime Components
    data['high_vol_vol_component'] = data['micro_vol'] * data['vol_efficiency']
    data['high_vol_fractal_component'] = data['fractal_change'] * data['gap_magnitude']
    data['high_vol_volume_component'] = data['volume_momentum'] * data['pressure_momentum']
    
    # Low Volatility Regime Components
    data['low_vol_vol_component'] = data['macro_vol'] * data['gap_fill_ratio']
    data['low_vol_fractal_component'] = data['hurst_short'] * data['volume_efficiency']
    data['low_vol_volume_component'] = data['fractal_volume'] * data['pressure_momentum']
    
    # Core Elasticity Factor Components
    data['vol_fractal_alignment'] = data['micro_vol'] * data['fractal_change']
    data['gap_pressure_confirmation'] = data['gap_fill_ratio'] * data['pressure_momentum']
    data['volume_fractal_consistency'] = data['volume_efficiency'] * data['hurst_short']
    
    # Regime-Adaptive Weighting
    def calculate_regime_weighted(row):
        if row['high_vol_regime']:
            return (0.45 * row['vol_fractal_alignment'] + 
                    0.30 * row['gap_pressure_confirmation'] + 
                    0.25 * row['volume_fractal_consistency'])
        elif row['low_vol_regime']:
            return (0.30 * row['vol_fractal_alignment'] + 
                    0.45 * row['gap_pressure_confirmation'] + 
                    0.25 * row['volume_fractal_consistency'])
        else:
            return (0.35 * row['vol_fractal_alignment'] + 
                    0.35 * row['gap_pressure_confirmation'] + 
                    0.30 * row['volume_fractal_consistency'])
    
    data['core_factor'] = data.apply(calculate_regime_weighted, axis=1)
    
    # Dynamic Enhancement
    data['volatility_elasticity'] = data['core_factor'] * (1 + data['vol_efficiency'])
    data['gap_momentum'] = data['core_factor'] * (1 + data['gap_magnitude'] * data['volume_momentum'])
    
    # Final Alpha
    data['final_alpha'] = data['core_factor'] * (1 + data['volatility_elasticity'] + data['gap_momentum'])
    
    return data['final_alpha']
