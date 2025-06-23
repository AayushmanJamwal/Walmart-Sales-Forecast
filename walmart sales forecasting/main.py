import pandas as pd
from src.features.build_features import create_features, scale_features
from src.models.train_model import train_random_forest, evaluate_model, save_model
from sklearn.model_selection import train_test_split

def main():
    # Load data
    df = pd.read_csv('data/raw/train.csv')
    
    # Feature engineering
    df = create_features(df)
    df = df.dropna()  # Remove rows with NA from lag features
    
    # Define features and target
    features = ['Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 
                'CPI', 'Unemployment', 'Year', 'Month', 'Week', 'Day',
                'Weekly_Sales_Lag1', 'Weekly_Sales_Rolling_Mean']
    target = 'Weekly_Sales'
    
    # Scale numerical features
    numerical_features = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
                         'Weekly_Sales_Lag1', 'Weekly_Sales_Rolling_Mean']
    df = scale_features(df, numerical_features)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[target], test_size=0.2, random_state=42
    )
    
    # Train model
    model = train_random_forest(X_train, y_train)
    
    # Evaluate
    rmse = evaluate_model(model, X_test, y_test)
    print(f"RMSE: {rmse}")
    
    # Save model
    save_model(model, 'models/random_forest_model.pkl')

if __name__ == "__main__":
    main()