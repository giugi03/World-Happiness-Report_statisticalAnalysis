import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt  
import sklearn 

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Carica il dataset
df = pd.read_csv("dataset.csv")

# Mostra le prime righe del dataset e le informazioni generali
print(df.head())
print(df.info())

# Controlla la presenza di valori NaN
print(df.isnull().sum())

# 1. Ripulire il dataset da eventuali NaN
df_cleaned = df.dropna()

# 2. Controllare che le variabili di tipo numerico non presentino dei valori fuori soglia
# Ottieni statistiche descrittive per identificare valori sospetti
numeric_data = df_cleaned.select_dtypes(include="number") #seleziona solo le colonne che contengono numeri
print(numeric_data.info())
print(numeric_data.describe()) #statistiche (media, quartili,...)

# Converte i valori infiniti in NaN e rimuovili
numeric_data.replace([np.inf, -np.inf], np.nan, inplace=True)
numeric_data.dropna(inplace=True)


# 3) EDA - Exploring Data Analysis
pairplot = sns.pairplot(numeric_data)
plt.show(pairplot)  # Mostra il pairplot

# Esplorare i 10 paesi più felici
top_countries = df_cleaned.nlargest(10, 'Score')

plt.figure(figsize=(10, 6))
sns.barplot(x=top_countries['Country or region'], y=top_countries['Score'])
plt.title('Top 10 Happiest Countries')
plt.xlabel('Country')
plt.ylabel('Happiness Score')
plt.xticks(rotation=45)
plt.show()

# Esplorare la distribuzione del punteggio di felicità
plt.figure(figsize=(10, 6))
sns.histplot(df_cleaned['Score'], bins=20, kde=True)
plt.title('Distribution of Happiness Scores')
plt.xlabel('Happiness Score')
plt.ylabel('Frequency')
plt.show()

# Esplorare i fattori che influenzano la felicità
plt.figure(figsize=(12, 8))
sns.boxplot(data=df_cleaned.drop(columns=['Overall rank', 'Country or region', 'Score']), orient='h', palette='Set2')
plt.title('Factors Influencing Happiness')
plt.xlabel('Value')
plt.show()

# Relazione tra GDP per capita e Score
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_cleaned, x='GDP per capita', y='Score', hue='Score', s=100)
plt.title('GDP per capita vs Happiness Score')
plt.xlabel('GDP per capita')
plt.ylabel('Happiness Score')
plt.show()

# Relazione tra Freedom to make life choices e Score
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_cleaned, x='Freedom to make life choices', y='Score', hue='Score', s=100)
plt.title('Freedom to make life choices vs Happiness Score')
plt.xlabel('Freedom to make life choices')
plt.ylabel('Happiness Score')
plt.show()

# Heatmap della matrice di correlazione
correlation_matrix = df_cleaned.drop(columns=['Overall rank', 'Country or region']).corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Matrix')
plt.show()

# 4) SPLITTING
from sklearn import model_selection

df_cleaned['Happiness_Level'] = pd.cut(df_cleaned['Score'], bins=3, labels=[0, 1, 2])  # Trasformare 'Score' in una variabile categorica
output_class = 'Happiness_Level'

# Definisco il seed per la riproducibilità
seed = 42

# Suddivido il dataset in training, validation e test set
data_train, data_test = model_selection.train_test_split(df_cleaned, train_size=108, random_state=seed)
data_train, data_val = model_selection.train_test_split(data_train, train_size=84, random_state=seed)

# Definisco le colonne di input e la colonna di output 
features = ['GDP per capita', 'Social support', 'Healthy life expectancy',
            'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']
output = 'Score'

# Suddivido X (input) e Y (output)
x_train = data_train[features]
Y_train = data_train[output]

x_val = data_val[features]
Y_val = data_val[output]

x_test = data_test[features]
Y_test = data_test[output]

# 5) REGRESSIONE
from sklearn.linear_model import LinearRegression

# Eseguire la regressione lineare tra le variabili fortemente correlate
x = df_cleaned['Social support'].values.reshape(-1, 1)
y = df_cleaned['GDP per capita'].values.reshape(-1, 1)

model = LinearRegression().fit(x, y)

# Stima dei coefficienti
coefficients = model.coef_
intercept = model.intercept_
print(f'Coefficiente: {coefficients[0]}')
print(f'Intercetta: {intercept}')

# Predizione
y_pred = model.predict(x)

# Grafico dei punti e della retta di regressione
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Dati')
plt.plot(x, y_pred, color='red', label='Retta di regressione')
plt.xlabel('Social support')
plt.ylabel('GDP per capita')
plt.title('Regressione Lineare tra Social support e GDP per capita')
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error, r2_score
# Calcolo del coefficiente r^2
r2 = r2_score(y, y_pred)
print(f'R^2: {r2}')

# Calcolo del MSE
mse = mean_squared_error(y, y_pred)
print(f'Mean Squared Error (MSE): {mse}')

import statsmodels.api as sm
# Analisi di normalità dei residui
residuals = y - y_pred

# Q-Q plot per l'analisi di normalità dei residui
plt.figure(figsize=(10, 6))
sm.qqplot(residuals, line ='45')
plt.title('Q-Q Plot dei Residui')
plt.xlabel('Quantili Teorici')
plt.ylabel('Quantili Campionari')
plt.show()

# Scatterplot dei residui
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Scatterplot dei Residui')
plt.xlabel('Valori Predetti')
plt.ylabel('Residui')
plt.show()

import scipy.stats as stats
# Test di normalità
stat, p_value = stats.shapiro(residuals)
print(f'Test di Shapiro-Wilk per la normalità dei residui: p-value = {p_value}')

# 6) ADDESTRAMENTO DEL MODELLO
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Suddivido X (input) e Y (output) per la classificazione
x_train_class = data_train[features]
y_train_class = data_train[output_class]

x_val_class = data_val[features]
y_val_class = data_val[output_class]

x_test_class = data_test[features]
y_test_class = data_test[output_class]

# Addestramento della Regressione Logistica
logistic_model = LogisticRegression()
logistic_model.fit(x_train_class, y_train_class)


# Addestramento della SVM
svm_model = SVC(kernel='linear')
svm_model.fit(x_train_class, y_train_class)


# 7) HYPERPARAMETER TUNING
best_degree = 0
best_accuracy = 0

print("\nHyperparameter Tuning for SVM")
for d in range(1, 11):
    model = SVC(kernel="poly", degree=d)
    model.fit(x_train_class, y_train_class)
    y_val_pred = model.predict(x_val_class)
    ME = np.sum(y_val_pred != y_val_class)
    MR = ME / len(y_val_pred)
    Acc = 1 - MR
    print(f"Accuracy for SVM with degree {d}: {Acc:.4f}")
    if Acc > best_accuracy:
        best_accuracy = Acc
        best_degree = d
        
print(f"Best degree for SVM: {best_degree}")

# Addestramento della SVM con il miglior grado trovato
svm_model_poly = SVC(kernel="poly", degree=best_degree)
svm_model_poly.fit(x_train_class, y_train_class)

# 8) VALUTAZIONE DELLA PERFORMANCE 
# Valutazione finale sul test set per la Regressione Logistica
y_pred_logistic = logistic_model.predict(x_test_class)
print("\nLogistic Regression on Test Set")
ME_logistic = np.sum(y_pred_logistic != y_test_class)
MR_logistic = ME_logistic / len(y_pred_logistic)
Acc_logistic = 1 - MR_logistic
print(f"Misclassification Error (Logistic Regression): {ME_logistic}")
print(f"Misclassification Rate (Logistic Regression): {MR_logistic}")
print(f"Accuracy (Logistic Regression): {Acc_logistic}")


# Valutazione finale sul test set per la SVM
y_pred_svm = svm_model.predict(x_test_class)
print("\nSVM on Test Set")
ME_svm = np.sum(y_pred_svm != y_test_class)
MR_svm = ME_svm / len(y_pred_svm)
Acc_svm = 1 - MR_svm
print(f"Misclassification Error (SVM): {ME_svm}")
print(f"Misclassification Rate (SVM): {MR_svm}")
print(f"Accuracy (SVM): {Acc_svm}")

# Valutazione finale sul test set per la SVM con il miglior grado trovato
y_pred_svm_poly = svm_model_poly.predict(x_test_class)
print("\nSVM with best degree on Test Set")
ME_svm = np.sum(y_pred_svm_poly != y_test_class)
MR_svm = ME_svm / len(y_pred_svm_poly)
Acc_svm = 1 - MR_svm
print(f"Misclassification Error (SVM): {ME_svm}")
print(f"Misclassification Rate (SVM): {MR_svm}")
print(f"Accuracy (SVM): {Acc_svm}")

# 9) STUDIO STATISTICO SUI RISULTATI DELLA VALUTAZIONE
from sklearn.metrics import accuracy_score

k = 10
svm_accuracies = []
logistic_accuracies = []

for i in range(k):
    data_train_val, data_test = model_selection.train_test_split(df_cleaned, train_size=0.85, random_state=seed + i)
    data_train, data_val = model_selection.train_test_split(data_train_val, train_size=(109/132), random_state=seed + i)

    X_train = data_train[features]
    y_train = data_train[output_class]
    X_test = data_test[features]
    y_test = data_test[output_class]

    # SVM
    svm_model_poly.fit(X_train, y_train)
    y_pred_svm = svm_model_poly.predict(X_test)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    svm_accuracies.append(accuracy_svm)
    
    # Regressione Logistica
    logistic_model.fit(X_train, y_train)
    y_pred_logistic = logistic_model.predict(X_test)
    accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
    logistic_accuracies.append(accuracy_logistic)

# Analisi statistica dei risultati per SVM
svm_accuracies = np.array(svm_accuracies)
mean_accuracy_svm = np.mean(svm_accuracies)
std_accuracy_svm = np.std(svm_accuracies)
conf_interval_svm = stats.norm.interval(0.95, loc=mean_accuracy_svm, scale=std_accuracy_svm / np.sqrt(k))

print(f"\nMean SVM Accuracy: {mean_accuracy_svm}")
print(f"Standard Deviation of SVM Accuracy: {std_accuracy_svm}")
print(f"95% Confidence Interval for SVM Accuracy: {conf_interval_svm}")

# Visualizzazione delle accuracies per SVM
plt.figure(figsize=(10, 6))
sns.histplot(svm_accuracies, bins=10, kde=True)
plt.title('Distribution of SVM Accuracies')
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x=svm_accuracies)
plt.title('Boxplot of SVM Accuracies')
plt.xlabel('Accuracy')
plt.show()

# Analisi statistica dei risultati per Logistic Regression
logistic_accuracies = np.array(logistic_accuracies)
mean_accuracy_logistic = np.mean(logistic_accuracies)
std_accuracy_logistic = np.std(logistic_accuracies)
conf_interval_logistic = stats.norm.interval(0.95, loc=mean_accuracy_logistic, scale=std_accuracy_logistic / np.sqrt(k))

print(f"\nMean Logistic Regression Accuracy: {mean_accuracy_logistic}")
print(f"Standard Deviation of Logistic Regression Accuracy: {std_accuracy_logistic}")
print(f"95% Confidence Interval for Logistic Regression Accuracy: {conf_interval_logistic}")

# Visualizzazione delle accuracies per Logistic Regression
plt.figure(figsize=(10, 6))
sns.histplot(logistic_accuracies, bins=10, kde=True)
plt.title('Distribution of Logistic Regression Accuracies')
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x=logistic_accuracies)
plt.title('Boxplot of Logistic Regression Accuracies')
plt.xlabel('Accuracy')
plt.show()
