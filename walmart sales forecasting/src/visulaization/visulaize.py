import matplotlib.pyplot as plt
import seaborn as sns

def plot_sales_trend(df, store_id):
    store_data = df[df['Store'] == store_id]
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Date', y='Weekly_Sales', data=store_data)
    plt.title(f'Sales Trend for Store {store_id}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'visualizations/store_{store_id}_trend.png')
    plt.close()

def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    indices = importance.argsort()[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.bar(range(len(indices)), importance[indices], align='center')
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig('visualizations/feature_importance.png')
    plt.close()