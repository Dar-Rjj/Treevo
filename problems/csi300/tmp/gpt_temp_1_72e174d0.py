import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility-Adjusted Momentum Components
    # Short-term (5-day)
    data['short_price_momentum'] = (data['close'] - data['close'].shift(5)) / (
        data['high'].rolling(window=5).apply(lambda x: np.sum(np.abs(x - data.loc[x.index, 'low'])), raw=False) / 5
    )
    data['short_volume_momentum'] = (data['volume'] - data['volume'].shift(5)) / (
        data['volume'].rolling(window=5).apply(lambda x: np.sum(np.abs(x - x.shift(1).fillna(0))), raw=False) / 5
    )
    data['short_combined_momentum'] = data['short_price_momentum'] * data['short_volume_momentum']
    
    # Medium-term (13-day)
    data['medium_price_momentum'] = (data['close'] - data['close'].shift(13)) / (
        data['high'].rolling(window=13).apply(lambda x: np.sum(np.abs(x - data.loc[x.index, 'low'])), raw=False) / 13
    )
    data['medium_volume_momentum'] = (data['volume'] - data['volume'].shift(13)) / (
        data['volume'].rolling(window=13).apply(lambda x: np.sum(np.abs(x - x.shift(1).fillna(0))), raw=False) / 13
    )
    data['medium_combined_momentum'] = data['medium_price_momentum'] * data['medium_volume_momentum']
    
    # Long-term (34-day)
    data['long_price_momentum'] = (data['close'] - data['close'].shift(34)) / (
        data['high'].rolling(window=34).apply(lambda x: np.sum(np.abs(x - data.loc[x.index, 'low'])), raw=False) / 34
    )
    data['long_volume_momentum'] = (data['volume'] - data['volume'].shift(34)) / (
        data['volume'].rolling(window=34).apply(lambda x: np.sum(np.abs(x - x.shift(1).fillna(0))), raw=False) / 34
    )
    data['long_combined_momentum'] = data['long_price_momentum'] * data['long_volume_momentum']
    
    # Volume-Price Efficiency Assessment
    data['price_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['volume_efficiency'] = data['volume'] / (data['amount'] / data['close']).replace(0, np.nan)
    data['intraday_efficiency'] = data['price_efficiency'] * data['volume_efficiency']
    
    # Multi-timeframe Efficiency Divergence
    data['short_divergence'] = data['short_combined_momentum'] / data['intraday_efficiency'].replace(0, np.nan)
    data['medium_divergence'] = data['medium_combined_momentum'] / data['intraday_efficiency'].replace(0, np.nan)
    data['long_divergence'] = data['long_combined_momentum'] / data['intraday_efficiency'].replace(0, np.nan)
    
    # Efficiency Momentum Persistence
    data['short_persistence'] = data['short_divergence'] / data['short_divergence'].shift(5).replace(0, np.nan)
    data['medium_persistence'] = data['medium_divergence'] / data['medium_divergence'].shift(8).replace(0, np.nan)
    data['long_persistence'] = data['long_divergence'] / data['long_divergence'].shift(13).replace(0, np.nan)
    
    # Volatility Breakout Integration
    data['short_breakout'] = ((data['close'] > data['high'].shift(1).rolling(window=4).max()) | 
                             (data['close'] < data['low'].shift(1).rolling(window=4).min())).astype(float)
    data['medium_breakout'] = ((data['close'] > data['high'].shift(1).rolling(window=12).max()) | 
                              (data['close'] < data['low'].shift(1).rolling(window=12).min())).astype(float)
    data['long_breakout'] = ((data['close'] > data['high'].shift(1).rolling(window=33).max()) | 
                            (data['close'] < data['low'].shift(1).rolling(window=33).min())).astype(float)
    
    # Volatility-Adjusted Breakout Strength
    data['short_breakout_strength'] = data['short_price_momentum']
    data['medium_breakout_strength'] = data['medium_price_momentum']
    data['long_breakout_strength'] = data['long_price_momentum']
    
    # Volume-Confirmed Volatility Breakouts
    data['short_volume_confirmation'] = (data['short_breakout'] * data['short_breakout_strength'] * 
                                       data['short_volume_momentum'])
    data['medium_volume_confirmation'] = (data['medium_breakout'] * data['medium_breakout_strength'] * 
                                        data['medium_volume_momentum'])
    data['long_volume_confirmation'] = (data['long_breakout'] * data['long_breakout_strength'] * 
                                      data['long_volume_momentum'])
    
    # Efficiency Momentum Framework
    data['opening_pressure'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['midday_pressure'] = (data['close'] - (data['open'] + data['high'] + data['low']) / 3) / (data['high'] - data['low']).replace(0, np.nan)
    data['intraday_pressure'] = data['opening_pressure'] * data['midday_pressure']
    
    # Multi-timeframe Momentum Alignment
    data['short_alignment'] = data['short_combined_momentum'] * data['intraday_pressure']
    data['medium_alignment'] = data['medium_combined_momentum'] * data['intraday_pressure']
    data['long_alignment'] = data['long_combined_momentum'] * data['intraday_pressure']
    
    # Persistence-Weighted Efficiency
    data['short_weighted'] = data['short_divergence'] * data['short_persistence']
    data['medium_weighted'] = data['medium_divergence'] * data['medium_persistence']
    data['long_weighted'] = data['long_divergence'] * data['long_persistence']
    
    # Adaptive Signal Synthesis
    # Cross-timeframe Efficiency Detection
    data['short_vs_medium'] = data['short_weighted'] / data['medium_weighted'].replace(0, np.nan)
    data['medium_vs_long'] = data['medium_weighted'] / data['long_weighted'].replace(0, np.nan)
    data['cross_timeframe_efficiency'] = (data['short_vs_medium'] + data['medium_vs_long']) / 2
    
    # Volume-Confirmed Volatility Signals
    data['short_confirmed'] = data['short_volume_confirmation'] * data['short_alignment']
    data['medium_confirmed'] = data['medium_volume_confirmation'] * data['medium_alignment']
    data['long_confirmed'] = data['long_volume_confirmation'] * data['long_alignment']
    data['volume_confirmed_volatility_signals'] = (data['short_confirmed'] + data['medium_confirmed'] + data['long_confirmed']) / 3
    
    # Multi-timeframe Signal Integration
    data['momentum_efficiency_base'] = data['cross_timeframe_efficiency'] * data['intraday_pressure']
    data['volume_volatility_enhancement'] = data['volume_confirmed_volatility_signals'] * (
        data['short_weighted'] + data['medium_weighted'] + data['long_weighted']) / 3
    data['breakout_confirmation'] = (data['short_volume_confirmation'] + data['medium_volume_confirmation'] + 
                                   data['long_volume_confirmation']) / 3 * (
        data['short_alignment'] + data['medium_alignment'] + data['long_alignment']) / 3
    
    # Final Alpha Generation
    # Core Signal Components
    data['volatility_adjusted_momentum_score'] = (data['short_alignment'] + data['medium_alignment'] + 
                                                data['long_alignment']) / 3
    data['volume_efficiency_momentum'] = data['cross_timeframe_efficiency'] * data['volume_confirmed_volatility_signals']
    data['breakout_volatility'] = (data['short_volume_confirmation'] + data['medium_volume_confirmation'] + 
                                 data['long_volume_confirmation']) / 3 * data['volatility_adjusted_momentum_score']
    
    # Non-linear Signal Enhancement with persistence weighting
    alpha_signal = (
        data['volatility_adjusted_momentum_score'] * 
        (1 + (data['short_persistence'] + data['medium_persistence'] + data['long_persistence']) / 3) *
        data['volume_efficiency_momentum'] *
        np.sign(data['breakout_volatility']) * np.sqrt(np.abs(data['breakout_volatility']))
    )
    
    return alpha_signal
