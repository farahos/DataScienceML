KMEANS_BUSINESS = {
    0: {
        "cluster_name": "Low-interest products",
        "description": "Low net quantity, very low revenue and few customers.",
        "recommended_action": "Promote, bundle, or consider discontinuation."
    },
    1: {
        "cluster_name": "Frequent low-price items",
        "description": "High number of transactions and customers but low revenue per item.",
        "recommended_action": "Cross-sell, consider small price increase or margin improvement."
    },
    2: {
        "cluster_name": "Best-sellers",
        "description": "Very high revenue, many customers and transactions.",
        "recommended_action": "Maintain stock, invest in promotion and visibility."
    },
    3: {
        "cluster_name": "High-volume / Discounted items",
        "description": "Very high quantity but low or negative revenue (likely discounted or high returns).",
        "recommended_action": "Review pricing and promotions; verify profitability."
    }
}

DBSCAN_BUSINESS = {
    -1: {
        "cluster_name": "Outliers / Special items",
        "description": "High revenue or transaction counts; seasonal, premium or niche products.",
        "recommended_action": "Stock selectively and run targeted seasonal promotions."
    },
    0: {
        "cluster_name": "Low-interest products",
        "description": "Low quantity and low revenue.",
        "recommended_action": "Promote or consider removing from catalog."
    },
    1: {
        "cluster_name": "Loss leaders",
        "description": "Very high quantity but negative or very low revenue — likely sold at a loss.",
        "recommended_action": "Optimize pricing and avoid overstocking; evaluate marketing role."
    },
    2: {
        "cluster_name": "Stable performers",
        "description": "Consistent sales with moderate revenue.",
        "recommended_action": "Keep steady stock levels; consider periodic promotions."
    },
    3: {
        "cluster_name": "High-margin niche items",
        "description": "Low quantity but high revenue per item (high margin).",
        "recommended_action": "Promote as premium products and focus targeted marketing."
    }
}