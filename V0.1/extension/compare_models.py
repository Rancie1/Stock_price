# Sentiment vs Baseline Model Comparison
# Compares the performance of sentiment-enhanced model vs baseline technical-only model

import pickle
import pandas as pd
import numpy as np
import logging
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_results():
    """Load results from both models."""
    try:
        # Load sentiment model results
        with open('../sentiment_data/classification_results.pkl', 'rb') as f:
            sentiment_results = pickle.load(f)
        logger.info("Loaded sentiment model results")
    except FileNotFoundError:
        logger.error("Sentiment model results not found. Run sentiment_classification.py first.")
        return None, None
    
    try:
        # Load baseline model results
        with open('../sentiment_data/baseline_results.pkl', 'rb') as f:
            baseline_results = pickle.load(f)
        logger.info("Loaded baseline model results")
    except FileNotFoundError:
        logger.error("Baseline model results not found. Run baseline_classification.py first.")
        return sentiment_results, None
    
    return sentiment_results, baseline_results


def compare_models():
    """Compare sentiment vs baseline model performance."""
    print("=" * 80)
    print("SENTIMENT vs BASELINE MODEL COMPARISON")
    print("=" * 80)
    print()
    
    # Load results
    sentiment_results, baseline_results = load_results()
    
    if sentiment_results is None:
        print("❌ Cannot load sentiment model results")
        return
    
    if baseline_results is None:
        print("❌ Cannot load baseline model results")
        return
    
    # Extract test data
    y_test_sentiment = sentiment_results['y_test']
    y_test_baseline = baseline_results['y_test']
    
    # Verify same test set
    if not np.array_equal(y_test_sentiment, y_test_baseline):
        logger.warning("Test sets are different between models")
    
    print("📊 MODEL PERFORMANCE COMPARISON")
    print("=" * 50)
    print()
    
    # Compare each model type
    model_types = ['Random Forest', 'Gradient Boosting', 'Logistic Regression', 'SVM']
    
    comparison_data = []
    
    for model_type in model_types:
        if model_type in sentiment_results['models'] and model_type in baseline_results['models']:
            sentiment_acc = sentiment_results['models'][model_type]['accuracy']
            baseline_acc = baseline_results['models'][model_type]['accuracy']
            
            sentiment_auc = sentiment_results['models'][model_type]['auc']
            baseline_auc = baseline_results['models'][model_type]['auc']
            
            acc_improvement = sentiment_acc - baseline_acc
            auc_improvement = sentiment_auc - baseline_auc
            
            comparison_data.append({
                'Model': model_type,
                'Baseline Accuracy': f"{baseline_acc:.3f}",
                'Sentiment Accuracy': f"{sentiment_acc:.3f}",
                'Accuracy Improvement': f"{acc_improvement:+.3f}",
                'Baseline AUC': f"{baseline_auc:.3f}",
                'Sentiment AUC': f"{sentiment_auc:.3f}",
                'AUC Improvement': f"{auc_improvement:+.3f}"
            })
    
    # Create comparison table
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    print()
    
    # Calculate average improvements
    avg_acc_improvement = np.mean([float(row['Accuracy Improvement']) for row in comparison_data])
    avg_auc_improvement = np.mean([float(row['AUC Improvement']) for row in comparison_data])
    
    print("📈 OVERALL IMPACT OF SENTIMENT ANALYSIS")
    print("=" * 50)
    print(f"Average Accuracy Improvement: {avg_acc_improvement:+.3f}")
    print(f"Average AUC Improvement: {avg_auc_improvement:+.3f}")
    print()
    
    # Determine best models
    sentiment_best = max(sentiment_results['models'].keys(), 
                        key=lambda k: sentiment_results['models'][k]['accuracy'])
    baseline_best = max(baseline_results['models'].keys(), 
                       key=lambda k: baseline_results['models'][k]['accuracy'])
    
    print("🏆 BEST MODEL COMPARISON")
    print("=" * 50)
    print(f"Best Baseline Model: {baseline_best}")
    print(f"  Accuracy: {baseline_results['models'][baseline_best]['accuracy']:.3f}")
    print(f"  AUC: {baseline_results['models'][baseline_best]['auc']:.3f}")
    print()
    print(f"Best Sentiment Model: {sentiment_best}")
    print(f"  Accuracy: {sentiment_results['models'][sentiment_best]['accuracy']:.3f}")
    print(f"  AUC: {sentiment_results['models'][sentiment_best]['auc']:.3f}")
    print()
    
    # Calculate improvement
    best_improvement = (sentiment_results['models'][sentiment_best]['accuracy'] - 
                       baseline_results['models'][baseline_best]['accuracy'])
    print(f"Best Model Accuracy Improvement: {best_improvement:+.3f}")
    print()
    
    # Feature importance comparison
    print("🔍 FEATURE IMPORTANCE COMPARISON")
    print("=" * 50)
    
    # Get feature names
    sentiment_features = sentiment_results.get('feature_names', [])
    baseline_features = baseline_results.get('feature_names', [])
    
    print(f"Sentiment Model Features: {len(sentiment_features)}")
    print(f"Baseline Model Features: {len(baseline_features)}")
    print(f"Sentiment Features Added: {len(sentiment_features) - len(baseline_features)}")
    print()
    
    # Check if sentiment features are in top features
    if hasattr(sentiment_results['best_model'], 'feature_importances_'):
        sentiment_importances = sentiment_results['best_model'].feature_importances_
        sentiment_feature_importance = list(zip(sentiment_features, sentiment_importances))
        sentiment_feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print("Top 10 Features in Sentiment Model:")
        for i, (feature, importance) in enumerate(sentiment_feature_importance[:10], 1):
            is_sentiment = any(sentiment_word in feature.lower() for sentiment_word in 
                             ['sentiment', 'article', 'news'])
            marker = "📰" if is_sentiment else "📊"
            print(f"  {i:2d}. {marker} {feature:<30} {importance:.4f}")
        print()
    
    # Summary
    print("📋 SUMMARY")
    print("=" * 50)
    if avg_acc_improvement > 0:
        print("✅ Sentiment analysis IMPROVES prediction accuracy")
        print(f"   Average improvement: {avg_acc_improvement:+.1%}")
    elif avg_acc_improvement < -0.01:
        print("❌ Sentiment analysis REDUCES prediction accuracy")
        print(f"   Average reduction: {avg_acc_improvement:+.1%}")
    else:
        print("➖ Sentiment analysis has MINIMAL impact on accuracy")
        print(f"   Average change: {avg_acc_improvement:+.1%}")
    
    if avg_auc_improvement > 0:
        print("✅ Sentiment analysis IMPROVES model discrimination")
    elif avg_auc_improvement < -0.01:
        print("❌ Sentiment analysis REDUCES model discrimination")
    else:
        print("➖ Sentiment analysis has MINIMAL impact on discrimination")
    
    print()
    print("✓ Comparison complete!")


if __name__ == "__main__":
    compare_models()
