import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Define file paths relative to the script location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE = os.path.join(BASE_DIR, "GiveMeSomeCredit-training.csv")
TEST_FILE = os.path.join(BASE_DIR, "GiveMeSomeCredit-testing.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "full_data.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "analysis_plots")

# Create plots directory if it doesn't exist
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_data():
    """Load train and test data."""
    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(f"Training file not found at {TRAIN_FILE}")
    if not os.path.exists(TEST_FILE):
        raise FileNotFoundError(f"Testing file not found at {TEST_FILE}")

    print("Loading data...")
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    return train_df, test_df

def preprocess_data(df):
    """Preprocess the data: handle missing values, drop duplicates."""
    # Drop Unnamed: 0 column if exists (it's likely an index from CSV)
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

    # Handle missing values
    # MonthlyIncome: Fill with median
    if 'MonthlyIncome' in df.columns:
        median_income = df['MonthlyIncome'].median()
        df['MonthlyIncome'] = df['MonthlyIncome'].fillna(median_income)
        
    # NumberOfDependents: Fill with mode (0 is common)
    if 'NumberOfDependents' in df.columns:
        df['NumberOfDependents'] = df['NumberOfDependents'].fillna(0)

    return df

def generate_statistics_report(df, name="Dataset"):
    """Generate and print descriptive statistics."""
    print(f"\n--- {name} Descriptive Statistics ---")
    desc = df.describe().T
    desc['skew'] = df.select_dtypes(include=[np.number]).skew()
    desc['kurtosis'] = df.select_dtypes(include=[np.number]).kurtosis()
    print(desc[['mean', 'std', 'min', '50%', 'max', 'skew', 'kurtosis']])

    # Missing values
    print(f"\n--- {name} Missing Values ---")
    print(df.isnull().sum()[df.isnull().sum() > 0])

def plot_distributions(df, name="Train"):
    """Plot distributions of numerical features."""
    print(f"\nGenerating distribution plots for {name}...")
    num_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in num_cols:
        if col == 'Unnamed: 0': continue
        
        plt.figure(figsize=(10, 5))
        
        # Histogram with KDE
        plt.subplot(1, 2, 1)
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f'{col} Distribution')
        
        # Boxplot
        plt.subplot(1, 2, 2)
        sns.boxplot(y=df[col])
        plt.title(f'{col} Boxplot')
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{col}_dist_{name}.png"))
        plt.close()

def plot_correlation_matrix(df, name="Train"):
    """Plot correlation matrix."""
    print(f"\nGenerating correlation matrix for {name}...")
    plt.figure(figsize=(12, 10))
    corr = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title(f'{name} Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"correlation_matrix_{name}.png"))
    plt.close()

def statistical_tests(df):
    """Perform statistical tests."""
    print("\n--- Statistical Tests (Train Data) ---")
    
    target = 'SeriousDlqin2yrs'
    if target not in df.columns:
        print("Target variable not found for statistical tests.")
        return

    # Compare features between Default (1) and Non-Default (0)
    print(f"\nComparing features by target: {target}")
    group0 = df[df[target] == 0]
    group1 = df[df[target] == 1]
    
    num_cols = df.select_dtypes(include=[np.number]).columns
    results = []
    
    for col in num_cols:
        if col == target or col == 'Unnamed: 0': continue
        
        # Mann-Whitney U Test (Non-parametric t-test equivalent)
        stat, p = stats.mannwhitneyu(group0[col], group1[col], alternative='two-sided')
        
        results.append({
            'Feature': col,
            'Group0_Mean': group0[col].mean(),
            'Group1_Mean': group1[col].mean(),
            'MW_Stat': stat,
            'P-Value': p,
            'Significant': p < 0.05
        })
    
    results_df = pd.DataFrame(results)
    print(results_df)
    
    # Save test results
    results_df.to_csv(os.path.join(BASE_DIR, "statistical_tests_results.csv"), index=False)

def main():
    try:
        train_df, test_df = load_data()
        
        # Preprocess
        train_processed = preprocess_data(train_df)
        train_processed['dataset'] = 'train'
        
        test_processed = preprocess_data(test_df)
        test_processed['dataset'] = 'test'
        
        # Statistics and Analysis on Train Data
        generate_statistics_report(train_processed, "Train")
        plot_distributions(train_processed, "Train")
        plot_correlation_matrix(train_processed, "Train")
        statistical_tests(train_processed)
        
        # Combine datasets
        print("\n--- Combining Datasets ---")
        full_df = pd.concat([train_processed, test_processed], axis=0, ignore_index=True)
        print(f"Full dataset shape: {full_df.shape}")
        
        # Save to single file
        full_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved full data to {OUTPUT_FILE}")
        print(f"Analysis plots saved to {PLOTS_DIR}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
