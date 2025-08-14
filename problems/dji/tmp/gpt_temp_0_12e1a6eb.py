importance for simplicity; in practice, these could be optimized
    weights = {'price_momentum': 0.4, 'volume_change': 0.3, 'volatility': 0.3}
    heuristics_matrix = (price_momentum * weights['price_momentum'] +
                        volume_change * weights['volume_change'] +
                        volatility * weights['volatility'])
    return heuristics_matrix
