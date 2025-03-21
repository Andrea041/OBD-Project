from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def apply_feature_selection(dataset, target):
    correlation_with_target = dataset.corr()[target].abs()

    correlation_with_target = correlation_with_target.drop(target)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=correlation_with_target.index, y=correlation_with_target.values, palette="coolwarm")
    plt.title(f"Correlation with Target: {target}")
    plt.xticks(rotation=45)
    plt.show()

    if target == 'class':
        threshold = 0.6
    else:
        threshold = 0.3

    features_to_keep = correlation_with_target[correlation_with_target > threshold].index.tolist()

    features_removed = [feature for feature in dataset.columns if feature not in features_to_keep + [target]]

    print(f"Features to keep: {features_to_keep}")
    print(f"Features removed: {features_removed}")

    dataset_selected = dataset[features_to_keep + [target]]

    return dataset_selected


# Function to balance classes
# oversample_threshold: Soglia minima per applicare SMOTE (default 0.3)
# undersample_threshold: Soglia massima per applicare undersampling (default 0.7)
def apply_balancing(X_train, y_train, oversample_threshold=0.3, undersample_threshold=0.7):
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
    dataset_original = dataset.drop_duplicates()

    # To test feature selection on dataset
    if feature_sel:
        # Print correlation heatmap for feature selection
        dataset_with_feature_selection = apply_feature_selection(dataset_original, target)
        X = dataset_with_feature_selection.drop(columns=[target])
    else:
        X = dataset_original.drop(columns=[target])
    y = dataset_original[target]

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=test_size, random_state=42)
    # Train-validation split
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, stratify=y_train, test_size=validation_size, random_state=42)

    if rebalancing:
        X_train, y_train = apply_balancing(X_train, y_train)

    return X_train, X_valid, X_test, y_train, y_valid, y_test