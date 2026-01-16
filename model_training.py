import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from mpl_toolkits.mplot3d import Axes3D

# Sklearn Tools
from sklearn.model_selection import train_test_split, cross_val_score 
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Machine Learning Algorithms (Criterion: Minimum 5 Models)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Global Plot Settings
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['savefig.dpi'] = 300 

# ==============================================================================
# CRISP-DM / KDD STEP 1 & 2: DATA ACQUISITION AND UNDERSTANDING
# ==============================================================================
print("\n--- STEP 1 & 2: DATA ACQUISITION AND UNDERSTANDING ---")

# Loading the dataset (SDSS DR19 - Released within the last 2 years)
df = pd.read_csv('Skyserver_SQL_SDSS_DR19.csv') 

# Dropping unique ID columns (KDD: Data Cleaning & Selection)
ids_to_drop = ['objid', 'specobjid']
df = df.drop(columns=ids_to_drop, errors='ignore')

print(f"Dataset Successfully Loaded. Shape: {df.shape}")

# ==============================================================================
# CRISP-DM / KDD STEP 3: DATA PREPARATION (Preprocessing & Feature Selection)
# ==============================================================================
print("\n--- STEP 3: DATA PREPARATION ---")

X = df.drop('class', axis=1)
y = df['class']

# Categorical Target Encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Splitting Data (Hold-out method: 80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Feature Selection using ANOVA F-test (Dimensionality Reduction)
print("Performing Feature Selection (ANOVA F-test)...")
selector = SelectKBest(score_func=f_classif, k=12) 
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Identifying Winning Features
cols_idxs = selector.get_support(indices=True)
selected_features = X.columns[cols_idxs].tolist()
print(f"Top 12 Selected Features: {selected_features}")

# Feature Scaling (Normalization for distance-based and gradient models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected) 

# ==============================================================================
# CRISP-DM / KDD STEP 4: MODELING & EVALUATION (Implementation of 5 Algorithms)
# ==============================================================================
print("\n--- STEP 4: MODELING & EVALUATION ---")

# Defining 5 different Machine Learning algorithms
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = []

for name, model in models.items():
    print(f"Evaluating {name}...")
    
    # 5-Fold Cross Validation (Methodological Diversity Criterion)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    mean_cv_acc = cv_scores.mean()
    
    # Standard Model Training
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # Performance Evaluation Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results.append([name, acc, mean_cv_acc, prec, rec, f1])
    print(f"-> Test Accuracy: {acc:.4f} | Mean CV Accuracy: {mean_cv_acc:.4f}")

# Storing Results in a DataFrame for Plotting
results_df = pd.DataFrame(
    results, 
    columns=["Algorithm", "Test_Accuracy", "CV_Mean_Accuracy", "Precision", "Recall", "F1-Score"]
)

# ==============================================================================
# VISUALIZATION: PERFORMANCE ANALYSIS
# ==============================================================================
print("\n--- EXPORTING PERFORMANCE PLOTS ---")

# PLOT 1: FEATURE IMPORTANCE (ANOVA SCORES)
scores = selector.scores_[cols_idxs]
plt.figure(figsize=(10, 8))
sorted_idx = np.argsort(scores)
plt.barh(np.array(selected_features)[sorted_idx], scores[sorted_idx], color='teal')
plt.xlabel("ANOVA F-Score")
plt.title("Feature Importance Ranking (KDD: Transformation)")
plt.tight_layout()
plt.savefig('1_feature_importance.png')
plt.close()

# PLOT 2: ACCURACY COMPARISON (TEST VS CROSS-VALIDATION)
results_melted = results_df.melt(
    id_vars="Algorithm", 
    value_vars=["Test_Accuracy", "CV_Mean_Accuracy"], 
    var_name="Metric", 
    value_name="Score"
)
plt.figure(figsize=(12, 6))
ax1 = sns.barplot(x="Score", y="Algorithm", hue="Metric", data=results_melted, palette="viridis")
plt.title('Model Performance Comparison (Hold-out vs 5-Fold CV)')
plt.xlim(0.90, 1.0) 
plt.tight_layout()
plt.savefig('2_accuracy_comparison.png')
plt.close()

# PLOT 3: ERROR ANALYSIS (CONFUSION MATRIX GRID)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()
class_names = le.classes_
cmap_options = ['Blues', 'Greens', 'Oranges', 'Purples', 'YlGnBu']

for i, (name, model) in enumerate(models.items()):
    y_pred_cm = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred_cm)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap_options[i], ax=axes[i],
                xticklabels=class_names, yticklabels=class_names, cbar=False)
    axes[i].set_title(f'{name} Confusion Matrix')
    axes[i].set_ylabel('Actual Label')
    axes[i].set_xlabel('Predicted Label')

# Remove the empty 6th subplot
if len(models) < 6:
    fig.delaxes(axes[5])

plt.tight_layout()
fig.suptitle("Error Analysis (Multi-Class Confusion Matrices)", fontsize=16, y=1.02)
plt.savefig('3_confusion_matrices.png')
plt.close()

# PLOT 4: DETAILED METRICS BY ALGORITHM
metrics_names = ["Test_Accuracy", "Precision", "Recall", "F1-Score"]
for i, (index, row) in enumerate(results_df.iterrows()):
    algo_name = row['Algorithm']
    values = [row['Test_Accuracy'], row['Precision'], row['Recall'], row['F1-Score']]
    plt.figure(figsize=(8, 6))
    bars = sns.barplot(x=metrics_names, y=values, palette='magma')
    for bar, val in zip(bars.patches, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002, 
                 f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    plt.title(f'{algo_name} Evaluation Metrics')
    plt.ylim(0.90, 1.01)
    plt.savefig(f'4_metrics_{algo_name.replace(" ", "_").lower()}.png')
    plt.close()

# ==============================================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
print("\n--- GENERATING EXPLORATORY DATA ANALYSIS (EDA) PLOTS ---")

# PLOT 5: CLASS DISTRIBUTION PIE CHART
class_counts = df['class'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=140, 
        colors=['#ff9999','#66b3ff','#99ff99'], explode=[0.05]*len(class_counts), shadow=True)
plt.title('Dataset Class Distribution (KDD: Data Understanding)')
plt.savefig('5_eda_class_distribution.png')
plt.close()

# PLOT 6: CORRELATION HEATMAP
numeric_cols = [col for col in selected_features if col in df.columns]
if len(numeric_cols) > 0:
    plt.figure(figsize=(12, 10))
    sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix of Selected Features')
    plt.tight_layout()
    plt.savefig('6_eda_correlation_heatmap.png')
    plt.close()

# PLOT 7: REDSHIFT DISTRIBUTION PER CLASS
if 'redshift' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='class', y='redshift', data=df, palette="Set2")
    plt.title('Redshift Distribution by Celestial Object Class')
    plt.savefig('7_eda_redshift_violin.png')
    plt.close()

# PLOT 8: 3D SCATTER PLOT (Color-Color-Redshift Separation)
if 'modelMag_u' in df.columns and 'modelMag_g' in df.columns:
    # Feature Engineering for visualization
    df['u_g'] = df['modelMag_u'] - df['modelMag_g']
    df['g_r'] = df['modelMag_g'] - df['modelMag_r']
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    colors_3d = {'STAR': 'blue', 'GALAXY': 'green', 'QSO': 'red'}
    
    for cls in df['class'].unique():
        # Subsampling (500 per class) to optimize 3D rendering
        subset = df[df['class'] == cls].sample(n=min(len(df[df['class']==cls]), 500), random_state=42)
        ax.scatter(subset['u_g'], subset['g_r'], subset['redshift'], 
                   c=colors_3d.get(cls, 'black'), label=cls, alpha=0.5)
    
    ax.set_xlabel('u-g Color Index')
    ax.set_ylabel('g-r Color Index')
    ax.set_zlabel('Redshift')
    ax.legend()
    plt.title('3D Geometric Separation of Classes')
    plt.savefig('8_eda_3d_scatter.png')
    plt.close()

print("\n--- ALL KDD/CRISP-DM STEPS COMPLETED. RESULTS SAVED SUCCESSFULLY ---")