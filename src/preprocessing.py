import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import numpy as np

def apply_feature_selection(X_train, target):
    original_type = type(X_train)

    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train)

    # Print correlation heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(X_train.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()

    corr_matrix = X_train.corr()
    drop_cols = set()

    # Correlation threshold
    threshold = 0.7

    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                if colname != target:
                    drop_cols.add(colname)

    print(f"\nFeatures to remove: {drop_cols}")
    print(f"Features removed: {len(drop_cols)}")

    X_train = X_train.drop(columns=drop_cols)

    if original_type is np.ndarray:
        return X_train.to_numpy()

    return X_train

# Function to balance classes
# oversample_threshold: Soglia minima per applicare SMOTE (default 0.3)
# undersample_threshold: Soglia massima per applicare undersampling (default 0.7)
def apply_balancing(X_train, y_train, dataset, target, oversample_threshold=0.3, undersample_threshold=0.7):
    sns.countplot(x = dataset[target])
    plt.title("Classes Distribution")
    plt.show()

    class_counts = Counter(y_train)
    total_samples = sum(class_counts.values())

    max_class = max(class_counts, key=class_counts.get)
    min_class = min(class_counts, key=class_counts.get)

    max_ratio = class_counts[max_class] / total_samples
    min_ratio = class_counts[min_class] / class_counts[max_class]

    print(f"\nClasses distribution: {class_counts}")

    if min_ratio < oversample_threshold:
        print("Applying oversampling\n")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    elif max_ratio > undersample_threshold:
        print("Applying undersampling\n")
        rus = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
    else:
        print("Classes already balanced, nothing to do!\n")
        return X_train, y_train

    print(f"\nDistribution after balancing: {Counter(y_resampled)}")
    return X_resampled, y_resampled

def show_encodings(column_name, label_encoders):
    if column_name in label_encoders:
        classes = label_encoders[column_name].classes_
        print(f"\nEncoding of '{column_name}':")
        for idx, val in enumerate(classes):
            print(f"{val} → {idx}")
    else:
        print(f"Column '{column_name}' not found")

# Function to encode dataset if there are categorical label
def encode_dataset(dataset, label):
    # Selezione delle colonne categoriche
    column_map = {
        "Air Quality": ['Temperature', 'Humidity', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Proximity_to_Industrial_Areas',
                        'Population_Density', 'Air Quality'],
        "class": ['objid', 'specobjid', 'ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'run', 'rerun', 'camcol', 'field', 'plate', 'mjd', 'fiberid', 'petroRad_u',
                  'petroRad_g', 'petroRad_i', 'petroRad_r', 'petroRad_z', 'petroFlux_u', 'petroFlux_g', 'petroFlux_i', 'petroFlux_r', 'petroFlux_z', 'petroR50_u',
                  'petroR50_g', 'petroR50_i', 'petroR50_r', 'petroR50_z', 'psfMag_u', 'psfMag_r', 'psfMag_g', 'psfMag_i', 'psfMag_z', 'expAB_u', 'expAB_g', 'expAB_r',
                  'expAB_i', 'expAB_z', 'redshift', 'class'],
        "Class": ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter',
                  'Extent', 'Solidity', 'roundness', 'Compactness', 'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4', 'Class'],
    }


    if label not in column_map:
        raise ValueError(f"Label '{label}' not recognized. Choose between: {list(column_map.keys())}")

    cat_cols = column_map[label]
    cat_cols = [col for col in cat_cols if col in dataset.columns]

    label_encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        dataset[col] = le.fit_transform(dataset[col])
        label_encoders[col] = le

    if label in label_encoders:
        show_encodings(label, label_encoders)

    return dataset, label_encoders

# Function to preprocess dataset
def preprocessing(dataset, target, test_size, validation_size, feature_sel, rebalancing):
    # Drop duplicates
    print(f"\nFound {dataset.duplicated().sum()} duplicated records!\n")
    dataset = dataset.drop_duplicates()

    # Split features and labels
    X = dataset.drop(columns=[target])
    y = dataset[target]

    print(f"Features number: {X.shape[1]}\n")

    # Remove records with Nan value in target column
    mask = y.notnull()
    X = X[mask]
    y = y[mask]

    # Substitute Nan values with median of the column
    if X.isnull().sum().any():
        X.fillna(X.median(), inplace=True)

    # Standardization
    ss = StandardScaler()
    X = ss.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    # Train-validation split
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=validation_size, random_state=42)

    # To test feature selection on dataset
    if feature_sel:
        # Print correlation heatmap for feature selection
        X_train = apply_feature_selection(X_train, target)

    if rebalancing:
        X_train, y_train = apply_balancing(X_train, y_train, dataset, target)

    return X_train, X_valid, X_test, y_train, y_valid, y_test