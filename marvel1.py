import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# Caminho relativo para o arquivo 'db.csv' dentro da pasta 'dados'
file_path = "./dados/db.csv"

# Carregar o arquivo CSV com o encoding 'cp1252'
df = pd.read_csv(file_path, encoding='cp1252')

# Remover espaços extras nos nomes das colunas
df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace(r'\s+', ' ', regex=True)

# Tratar valores ausentes apenas nas colunas numéricas
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Filtrar dados da Marvel e DC
marvel_df = df[df['Company'] == 'Marvel']
dc_df = df[df['Company'] == 'DC']

# Separar recursos e target
X_marvel = marvel_df[['Rate', 'Metascore', 'Minutes', 'Budget']]
y_marvel = marvel_df['Gross Worldwide']
X_dc = dc_df[['Rate', 'Metascore', 'Minutes', 'Budget']]
y_dc = dc_df['Gross Worldwide']

# Normalização dos dados
scaler = StandardScaler()
X_marvel_scaled = scaler.fit_transform(X_marvel)
X_dc_scaled = scaler.fit_transform(X_dc)

# Dividir os dados em conjuntos de treino e teste
X_train_marvel, X_test_marvel, y_train_marvel, y_test_marvel = train_test_split(X_marvel_scaled, y_marvel, test_size=0.2, random_state=42)
X_train_dc, X_test_dc, y_train_dc, y_test_dc = train_test_split(X_dc_scaled, y_dc, test_size=0.2, random_state=42)

# Modelo Lasso com ajuste de hiperparâmetros
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10]}  # Testar valores menores para alpha
lasso_model_marvel = GridSearchCV(Lasso(max_iter=10000), param_grid, scoring='neg_mean_absolute_error', cv=5)
lasso_model_dc = GridSearchCV(Lasso(max_iter=10000), param_grid, scoring='neg_mean_absolute_error', cv=5)

# Treinamento
lasso_model_marvel.fit(X_train_marvel, y_train_marvel)
lasso_model_dc.fit(X_train_dc, y_train_dc)

# Previsões
lasso_preds_marvel = lasso_model_marvel.predict(X_test_marvel)
lasso_preds_dc = lasso_model_dc.predict(X_test_dc)

# Cálculo do MAE
mae_lasso_marvel = mean_absolute_error(y_test_marvel, lasso_preds_marvel)
mae_lasso_dc = mean_absolute_error(y_test_dc, lasso_preds_dc)

# Resultados
print(f'MAE Lasso - Marvel: {mae_lasso_marvel:.2f}')
print(f'MAE Lasso - DC: {mae_lasso_dc:.2f}')

# Gráficos de Previsão
plt.figure(figsize=(18, 6))

# Gráfico 1: Lasso Predictions for Marvel
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test_marvel, y=lasso_preds_marvel, color='red', label='Marvel Lasso', alpha=0.6)
plt.plot([0, 1e9], [0, 1e9], 'k--')  # linha tracejada
plt.title('Previsões Lasso - Marvel')
plt.xlabel('Bilheteira Real (R$)')
plt.ylabel('Bilheteira Prevista (R$)')
plt.legend()
plt.grid()
plt.annotate(f'MAE: {mae_lasso_marvel:.2f}', xy=(0.7 * max(y_test_marvel), 0.3 * max(lasso_preds_marvel)), fontsize=12, color='red')

# Gráfico 2: Lasso Predictions for DC
plt.subplot(1, 2, 2)
sns.scatterplot(x=y_test_dc, y=lasso_preds_dc, color='blue', label='DC Lasso', alpha=0.6)
plt.plot([0, 1e9], [0, 1e9], 'k--')  # linha tracejada
plt.title('Previsões Lasso - DC')
plt.xlabel('Bilheteira Real (R$)')
plt.ylabel('Bilheteira Prevista (R$)')
plt.legend()
plt.grid()
plt.annotate(f'MAE: {mae_lasso_dc:.2f}', xy=(0.7 * max(y_test_dc), 0.3 * max(lasso_preds_dc)), fontsize=12, color='blue')

plt.tight_layout()
plt.show()

# Gráfico 3: Comparativo entre Marvel e DC
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test_marvel, y=lasso_preds_marvel, color='red', label='Marvel Lasso', alpha=0.6)
sns.scatterplot(x=y_test_dc, y=lasso_preds_dc, color='blue', label='DC Lasso', alpha=0.6)
plt.plot([0, 1e9], [0, 1e9], 'k--')  # linha tracejada
plt.title('Comparativo de Previsões Lasso - Marvel vs DC')
plt.xlabel('Bilheteira Real (R$)')
plt.ylabel('Bilheteira Prevista (R$)')
plt.legend()
plt.grid()
plt.show()
