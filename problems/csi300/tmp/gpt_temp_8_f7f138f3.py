import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original dataframe
    data = df.copy()
    
    # Price-Volume Divergence Analysis
    # Short-term divergence
    data['price_change_t'] = (data['close'] - data['open']) / (data['volume'] + 1e-8)
    data['price_change_t_1'] = data['price_change_t'].shift(1)
    data['short_term_divergence'] = data['price_change_t'] - data['price_change_t_1']
    
    # Medium-term divergence
    data['price_5d'] = data['close'].pct_change(5)
    data['volume_5d'] = data['volume'].rolling(5).sum()
    data['price_10d'] = data['close'].pct_change(10)
    data['volume_10d'] = data['volume'].rolling(10).sum()
    
    data['medium_term_ratio_5d'] = data['price_5d'] / (data['volume_5d'] + 1e-8)
    data['medium_term_ratio_10d'] = data['price_10d'] / (data['volume_10d'] + 1e-8)
    data['medium_term_divergence'] = data['medium_term_ratio_5d'] - data['medium_term_ratio_10d']
    
    # Divergence persistence
    data['divergence_direction'] = np.sign(data['short_term_divergence'])
    data['divergence_persistence'] = data['divergence_direction'].rolling(3).apply(
        lambda x: len(x[x == x.iloc[-1]]) if len(x) == 3 else np.nan, raw=False
    )
    
    # Market Microstructure Signals
    # Bid-ask spread proxy
    data['bid_ask_spread_proxy'] = (data['high'] - data['low']) / (data['close'] + 1e-8)
    
    # Price impact efficiency
    data['price_impact_efficiency'] = np.abs(data['close'] - data['open']) / (
        (data['volume'] + 1e-8) * (data['high'] - data['low'] + 1e-8)
    )
    
    # Trade size distribution (using price range as proxy for number of trades)
    data['trade_size_distribution'] = data['volume'] / (data['high'] - data['low'] + 1e-8)
    
    # Multi-Timeframe Momentum Patterns
    # Momentum convergence
    data['momentum_2d'] = data['close'] / data['close'].shift(2) - 1
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_8d'] = data['close'] / data['close'].shift(8) - 1
    data['momentum_convergence'] = data['momentum_2d'] + data['momentum_5d'] + data['momentum_8d']
    
    # Momentum divergence
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_6d'] = data['close'] / data['close'].shift(6) - 1
    data['momentum_9d'] = data['close'] / data['close'].shift(9) - 1
    data['momentum_divergence'] = data['momentum_3d'] - data['momentum_6d'] - data['momentum_9d']
    
    # Momentum stability
    data['daily_returns'] = data['close'].pct_change()
    data['momentum_stability_3d'] = data['daily_returns'].rolling(3).std()
    data['momentum_stability_8d'] = data['daily_returns'].rolling(8).std()
    data['momentum_stability'] = data['momentum_stability_3d'] / (data['momentum_stability_8d'] + 1e-8)
    
    # Alpha Factors
    # Divergence-Momentum Composite
    data['divergence_momentum_composite'] = (
        data['short_term_divergence'] * 
        data['momentum_convergence'] * 
        data['price_impact_efficiency'] * 
        data['divergence_persistence']
    )
    
    # Microstructure Momentum
    data['microstructure_momentum'] = (
        data['bid_ask_spread_proxy'] * 
        data['momentum_divergence'] * 
        data['trade_size_distribution']
    )
    
    # Stability-Divergence Alpha
    data['stability_divergence_alpha'] = (
        data['momentum_stability'] * 
        data['medium_term_divergence'] * 
        data['price_impact_efficiency'] * 
        data['bid_ask_spread_proxy']
    )
    
    # Final alpha factor - weighted combination of all three
    final_alpha = (
        0.4 * data['divergence_momentum_composite'] +
        0.3 * data['microstructure_momentum'] +
        0.3 * data['stability_divergence_alpha']
    )
    
    return final_alpha
