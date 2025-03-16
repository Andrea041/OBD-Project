import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd

def apply_feature_selection(dataset):
    # Print correlation heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()

    corr_matrix = dataset.corr()
    drop_cols = set()

    # Correlation threshold
    threshold = 0.6

    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                drop_cols.add(colname)

    print(f"\nFeatures to remove: {drop_cols}")
    return dataset.drop(columns=drop_cols)

# Function to balance classes
# oversample_threshold: Soglia minima per applicare SMOTE (default 0.3)
# undersample_threshold: Soglia massima per applicare undersampling (default 0.7)
def apply_balancing(dataset, target, oversample_threshold=0.3, undersample_threshold=0.7):
    sns.countplot(x = dataset[target])
    plt.title("Classes Distribution")
    plt.show()

    X = dataset.drop(columns=[target])
    y = dataset[target]

    class_counts = Counter(y)
    total_samples = sum(class_counts.values())

    max_class = max(class_counts, key=class_counts.get)
    min_class = min(class_counts, key=class_counts.get)

    max_ratio = class_counts[max_class] / total_samples
    min_ratio = class_counts[min_class] / class_counts[max_class]

    print(f"\nClasses distribution: {class_counts}")

    if min_ratio < oversample_threshold:
        print("Applying oversampling\n")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

    elif max_ratio > undersample_threshold:
        print("Applying undersampling\n")
        rus = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = rus.fit_resample(X, y)

    else:
        print("Classes already balanced, nothing to do!\n")
        return dataset

    dataset_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    dataset_resampled[target] = y_resampled

    print(f"\nDistribution after balancing: {Counter(dataset_resampled[target])}")
    return dataset_resampled

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
        "NObeyesdad": ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE',
                       'SCC', 'CALC', 'MTRANS', 'NObeyesdad'],
        "Failure Type": ['UDI', 'Product ID', 'Type', 'Air temperature [K]', 'Process temperature [K]',
                         'Rotational speed [rpm]',
                         'Torque [Nm]', 'Tool wear [min]', 'Target', 'Failure Type'],
        "Var_1": ['ID', 'Gender', 'Ever_Married', 'Age', 'Graduated', 'Profession', 'Work_Experience', 'Spending_Score',
                  'Family_Size', 'Var_1', 'Segmentation'],
        "Air Quality": ['Temperature', 'Humidity', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Proximity_to_Industrial_Areas',
                        'Population_Density', 'Air Quality'],
        "class": ['objid', 'specobjid', 'ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'run', 'rerun', 'camcol', 'field', 'plate', 'mjd', 'fiberid', 'petroRad_u',
                  'petroRad_g', 'petroRad_i', 'petroRad_r', 'petroRad_z', 'petroFlux_u', 'petroFlux_g', 'petroFlux_i', 'petroFlux_r', 'petroFlux_z', 'petroR50_u',
                  'petroR50_g', 'petroR50_i', 'petroR50_r', 'petroR50_z', 'psfMag_u', 'psfMag_r', 'psfMag_g', 'psfMag_i', 'psfMag_z', 'expAB_u', 'expAB_g', 'expAB_r',
                  'expAB_i', 'expAB_z', 'redshift', 'class'],
        "Class": ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter',
                  'Extent', 'Solidity', 'roundness', 'Compactness', 'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4', 'Class'],
        "Label": ['','Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol', 'Timestamp', 'Flow Duration', 'Total Fwd Packets',
                   'Total Backward Packets','Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Fwd Packet Length Max', 'Fwd Packet Length Min',
                   'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
                   'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total',
                   'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags',
                   'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s',' Bwd Packets/s', 'Min Packet Length',
                   'Max Packet Length', 'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count',
                   'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size', 'Avg Fwd Segment Size',
                   'Avg Bwd Segment Size', 'Fwd Header Length.1', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk',
                   'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'Init_Win_bytes_forward',
                   'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward', 'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Label']
    }

    # Verifica se il label è valido
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

# Function to encode dataset if label are numerical
def encode_dataset_numerical(dataset, label):
    cat_cols = dataset.select_dtypes(include=['object']).columns.tolist()

    label_encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        dataset[col] = le.fit_transform(dataset[col])
        label_encoders[col] = le

    # Verifica se la colonna target è numerica
    if pd.api.types.is_numeric_dtype(dataset[label]):
        print(f"La colonna target '{label}' è numerica e non verrà codificata.")
    else:
        # Se non è numerica, la codifichiamo
        le = LabelEncoder()
        dataset[label] = le.fit_transform(dataset[label])
        label_encoders[label] = le

    if label in label_encoders:
        show_encodings(label, label_encoders)

    return dataset, label_encoders


# Function to preprocess dataset
def preprocessing(dataset, target, test_size, validation_size, feature_sel, rebalancing):
    # To test feature selection on dataset
    if feature_sel:
        # Print correlation heatmap for feature selection
        dataset = apply_feature_selection(dataset)

    if rebalancing:
        dataset = apply_balancing(dataset, target)

    # Drop duplicates
    print(f"\nFound {dataset.duplicated().sum()} duplicated records!\n")
    dataset = dataset.drop_duplicates()

    # Split features and labels
    X = dataset.drop(columns=[target])
    y = dataset[target]

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

    return X_train, X_valid, X_test, y_train, y_valid, y_test