from sklearn import feature_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

def correlation_heatmap(dataset):
    plt.figure(figsize=(10, 6))
    sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()

def apply_feature_selection(dataset):
    corr_matrix = dataset.corr()
    drop_cols = set()

    # Correlation threshold
    threshold = 0.6

    # Scansiona la matrice di correlazione
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                drop_cols.add(colname)

    print(f"\nFeatures to remove: {drop_cols}")
    return dataset.drop(columns=drop_cols)

# Function to encode Obesity dataset
def encode_obesity_dataset(dataset):
    # Select categorical columns
    cat_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE',
                'SCC', 'CALC', 'MTRANS', 'NObeyesdad']

    # Apply Label Encoding
    LabelEncoder()
    # Create a dictionary to store the encoders
    label_encoders = {}

    for col in cat_cols:
        label_encoders[col] = LabelEncoder()
        dataset[col] = label_encoders[col].fit_transform(dataset[col])

    # To see the mappings:
    def show_encodings(column_name):
        classes = label_encoders[column_name].classes_
        encoded = range(len(classes))
        print("\nEncoding of " + column_name + ":")
        for original, encoded_val in zip(classes, encoded):
            print(f"{original} - {encoded_val}")

    show_encodings('NObeyesdad')

# Function to preprocess dataset
def preprocessing(dataset, target, test_size, validation_size, feature_sel):
    # To test feature selection on dataset
    if feature_sel:
        # Print correlation heatmap for feature selection
        correlation_heatmap(dataset)
        dataset = apply_feature_selection(dataset)

    # Drop duplicates
    print(f"\nFound {dataset.duplicated().sum()} duplicated records!\n")
    dataset = dataset.drop_duplicates()

    # Split features and labels
    X = dataset.drop(columns = [target])
    y = dataset[target]

    # Remove records with Nan value in target column
    mask = y.notnull()
    X = X[mask]
    y = y[mask]

    # Substitute Nan values with median of the column
    if X.isnull().sum().any():
        X.fillna(X.median(), inplace = True)

    # Standardization
    ss = StandardScaler()
    X = ss.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 42)
    # Train-validation split
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = validation_size, random_state = 42)

    return X_train, X_valid, X_test, y_train, y_valid, y_test