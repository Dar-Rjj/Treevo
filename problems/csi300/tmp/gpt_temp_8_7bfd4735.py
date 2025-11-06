import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Price Fractals
    data['short_term_range'] = (data['high'] - data['low']) / data['close']
    data['medium_term_range'] = (data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min()) / data['close'].shift(4)
    data['long_term_range'] = (data['high'].rolling(window=10).max() - data['low'].rolling(window=10).min()) / data['close'].shift(9)
    data['fractal_compression_ratio'] = data['short_term_range'] / data['medium_term_range']
    
    # Volume Asymmetry Patterns
    data['price_change'] = data['close'] - data['close'].shift(1)
    data['up_day_volume'] = np.where(data['price_change'] > 0, data['volume'], np.nan)
    data['down_day_volume'] = np.where(data['price_change'] < 0, data['volume'], np.nan)
    
    up_volume_rolling = data['up_day_volume'].rolling(window=5, min_periods=1).mean()
    down_volume_rolling = data['down_day_volume'].rolling(window=5, min_periods=1).mean()
    data['volume_asymmetry_ratio'] = up_volume_rolling / down_volume_rolling
    data['volume_compression'] = data['volume'] / data['volume'].shift(4)
    
    # Price-Volume Fractal Alignment
    data['fractal_expansion'] = (data['fractal_compression_ratio'] > 1.2).astype(int)
    data['fractal_contraction'] = (data['fractal_compression_ratio'] < 0.8).astype(int)
    data['volume_fractal_momentum'] = data['volume_compression'] * data['fractal_compression_ratio']
    data['pv_fractal_divergence'] = np.sign(data['price_change']) * np.sign(data['volume_compression'] - 1)
    
    # Bid-Ask Imbalance Microstructure
    data['mid_price'] = (data['high'] + data['low']) / 2
    data['effective_spread'] = abs(data['close'] - data['mid_price']) / data['mid_price']
    data['relative_spread_momentum'] = data['effective_spread'] / data['effective_spread'].shift(1)
    data['spread_volume_efficiency'] = data['effective_spread'] * data['volume']
    
    historical_range = (data['high'].shift(4) - data['low'].shift(4)) / data['close'].shift(4)
    data['spread_compression'] = data['effective_spread'] / historical_range
    
    # Trade Size Distribution
    data['avg_trade_size'] = data['amount'] / data['volume']
    historical_avg_size = data['avg_trade_size'].shift(4)
    
    def count_large_trades(window):
        return np.sum(window > 2 * historical_avg_size.loc[window.index[-1]]) / len(window)
    
    def count_small_trades(window):
        return np.sum(window < 0.5 * historical_avg_size.loc[window.index[-1]]) / len(window)
    
    data['large_trade_concentration'] = data['avg_trade_size'].rolling(window=3).apply(count_large_trades, raw=False)
    data['small_trade_dominance'] = data['avg_trade_size'].rolling(window=3).apply(count_small_trades, raw=False)
    data['trade_size_skew'] = data['large_trade_concentration'] - data['small_trade_dominance']
    data['trade_size_volatility'] = data['avg_trade_size'].rolling(window=5).std()
    
    # Microstructure Momentum
    data['spread_efficiency_momentum'] = data['spread_volume_efficiency'] / data['spread_volume_efficiency'].shift(1)
    data['trade_size_momentum'] = data['trade_size_skew'] * data['volume_compression']
    data['microstructure_alignment'] = np.sign(data['effective_spread']) * np.sign(data['trade_size_skew'])
    
    high_low_range = data['high'] - data['low']
    data['bid_ask_pressure'] = ((data['close'] - data['low']) / high_low_range * data['volume'] - 
                               (data['high'] - data['close']) / high_low_range * data['volume'])
    
    # Multi-Scale Momentum Divergence
    data['short_momentum'] = data['close'] / data['close'].shift(1) - 1
    data['medium_momentum'] = data['close'] / data['close'].shift(4) - 1
    data['long_momentum'] = data['close'] / data['close'].shift(9) - 1
    data['momentum_compression'] = data['short_momentum'] / data['medium_momentum']
    
    data['volume_weighted_short_momentum'] = data['short_momentum'] * data['volume']
    data['volume_weighted_medium_momentum'] = data['medium_momentum'] * data['volume'].shift(4)
    data['volume_weighted_long_momentum'] = data['long_momentum'] * data['volume'].shift(9)
    data['volume_momentum_divergence'] = data['volume_weighted_short_momentum'] - data['volume_weighted_medium_momentum']
    
    # Fractal Breakout Signals
    data['fractal_expansion_breakout'] = np.where(data['fractal_compression_ratio'] > 1.2, data['short_momentum'], 0)
    data['fractal_contraction_reversal'] = np.where(data['fractal_compression_ratio'] < 0.8, data['short_momentum'], 0)
    data['volume_confirmed_breakout'] = data['fractal_expansion_breakout'] * data['volume_asymmetry_ratio']
    data['microstructure_breakout'] = data['fractal_expansion_breakout'] * data['trade_size_skew']
    
    # Price-Volume Efficiency Metrics
    data['intraday_efficiency'] = abs(data['close'] - data['open']) / (data['high'] - data['low'])
    data['overnight_efficiency'] = abs(data['open'] - data['close'].shift(1)) / (data['high'] - data['low'])
    data['total_range_efficiency'] = (data['high'] - data['low']) / abs(data['close'].shift(1) - data['open'])
    data['efficiency_compression'] = data['intraday_efficiency'] / data['overnight_efficiency']
    
    data['volume_per_price_unit'] = data['volume'] / (data['high'] - data['low'])
    data['volume_efficiency_momentum'] = data['volume_per_price_unit'] / data['volume_per_price_unit'].shift(1)
    data['efficiency_volume_alignment'] = np.sign(data['intraday_efficiency'] - 0.5) * np.sign(data['volume_efficiency_momentum'] - 1)
    
    historical_volume_per_unit = data['volume'].shift(4) / (data['high'].shift(4) - data['low'].shift(4))
    data['volume_range_compression'] = data['volume_per_price_unit'] / historical_volume_per_unit
    
    # Multi-Scale Efficiency Divergence
    data['short_long_efficiency_divergence'] = data['intraday_efficiency'] - data['total_range_efficiency']
    data['volume_efficiency_momentum_divergence'] = data['volume_efficiency_momentum'] * data['efficiency_compression']
    data['range_volume_alignment'] = np.sign(data['intraday_efficiency']) * np.sign(data['volume_per_price_unit'])
    data['fractal_efficiency_signal'] = data['efficiency_compression'] * data['fractal_compression_ratio']
    
    # Dynamic Alpha Synthesis
    # Core Asymmetry Components
    data['pv_fractal_alpha'] = data['pv_fractal_divergence'] * data['volume_fractal_momentum']
    data['microstructure_asymmetry_alpha'] = data['bid_ask_pressure'] * data['trade_size_skew']
    data['momentum_fractal_alpha'] = data['volume_momentum_divergence'] * data['momentum_compression']
    data['efficiency_asymmetry_alpha'] = data['volume_efficiency_momentum_divergence'] * data['range_volume_alignment']
    
    # Multi-Scale Confirmation
    data['fractal_expansion_weight'] = data['fractal_compression_ratio']
    data['volume_asymmetry_weight'] = data['volume_asymmetry_ratio']
    
    # Validated Asymmetry Signals
    data['fractal_volume_alpha'] = data['pv_fractal_alpha'] * data['volume_asymmetry_weight']
    data['microstructure_momentum_alpha'] = data['microstructure_asymmetry_alpha'] * data['momentum_compression']
    data['volume_efficiency_alpha'] = data['efficiency_asymmetry_alpha'] * data['volume_efficiency_momentum']
    data['fractal_breakout_alpha'] = data['volume_confirmed_breakout'] * data['fractal_expansion_weight']
    
    # Final Alpha Construction
    primary_factor = data['fractal_volume_alpha'] * data['trade_size_skew']
    secondary_factor = data['microstructure_momentum_alpha'] * data['spread_efficiency_momentum']
    tertiary_factor = data['volume_efficiency_alpha'] * data['efficiency_volume_alignment']
    quaternary_factor = data['fractal_breakout_alpha'] * data['fractal_efficiency_signal']
    
    # Composite Alpha with multi-scale confirmation
    composite_alpha = (primary_factor * 0.4 + 
                      secondary_factor * 0.3 + 
                      tertiary_factor * 0.2 + 
                      quaternary_factor * 0.1)
    
    return composite_alpha
