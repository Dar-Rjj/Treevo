import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility Regime Identification
    data['daily_volatility'] = (data['high'] - data['low']) / data['close']
    data['volatility_rolling'] = data['daily_volatility'].rolling(window=20, min_periods=10).std()
    
    # Calculate volatility quintiles
    data['volatility_quantile'] = data['volatility_rolling'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.qcut(x, q=5, labels=False, duplicates='drop').iloc[-1] if len(x.dropna()) >= 10 else np.nan,
        raw=False
    )
    
    # Momentum Quality Assessment
    data['momentum_5d'] = data['close'] / data['close'].shift(5)
    data['momentum_20d'] = data['close'] / data['close'].shift(20)
    data['acceleration'] = data['momentum_5d'] - data['momentum_20d']
    data['efficiency'] = data['amount'] / data['volume']
    data['quality_score'] = data['acceleration'] * data['efficiency']
    
    # Volume-Price Alignment
    data['intraday_pressure'] = (data['close'] - data['open']) * data['volume']
    
    # Calculate turnover components
    data['high_turnover'] = data['high'] * data['volume']
    data['low_turnover'] = data['low'] * data['volume']
    data['high_turnover_ratio'] = data['high_turnover'] / data['high_turnover'].shift(5)
    data['low_turnover_ratio'] = data['low_turnover'] / data['low_turnover'].shift(5)
    data['turnover_divergence'] = data['high_turnover_ratio'] - data['low_turnover_ratio']
    
    data['alignment_score'] = data['intraday_pressure'] * data['turnover_divergence']
    
    # Regime-Adaptive Weighting
    def get_regime_weight(quantile):
        if pd.isna(quantile):
            return 0.5
        if quantile >= 4:  # High volatility (top quintile)
            return 0.7
        elif quantile <= 1:  # Low volatility (bottom quintile)
            return 0.3
        else:  # Medium volatility
            return 0.5
    
    data['regime_weight'] = data['volatility_quantile'].apply(get_regime_weight)
    
    # Final Alpha Factor
    data['alpha_factor'] = (
        data['regime_weight'] * data['quality_score'] + 
        (1 - data['regime_weight']) * data['alignment_score']
    )
    
    # Return the alpha factor series
    return data['alpha_factor']
