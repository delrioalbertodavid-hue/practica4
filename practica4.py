import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Carga del dataset
dataset = pd.read_csv("superstore_dataset2012.csv")

# Explora y prepara los datos
# Realiza una exploración inicial del dataset para entender su estructura.
# Verifica los tipos de datos, valores nulos y realiza las transformaciones necesarias 
# (como convertir fechas al formato adecuado).
dataset.info()
dataset.isnull().sum()
dataset.dropna(inplace=True)
dataset['Order Date'] = pd.to_datetime(dataset['Order Date'], errors='coerce', dayfirst=True)
dataset['Ship Date'] = pd.to_datetime(dataset['Ship Date'], errors='coerce', dayfirst=True)
dataset.head()

# Crea visualizaciones univariantes con Matplotlib
# Implementa al menos un histograma o diagrama de barras utilizando Matplotlib para 
# visualizar la distribución de una variable numérica (como Ventas o Beneficios) o la 
# frecuencia de una variable categórica (como Categoría o Segmento).

plt.hist(dataset['Sales'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribución de Ventas')
plt.xlabel('Ventas')
plt.ylabel('Frecuencia')
plt.show()


sns.countplot(x="Category", data=dataset, palette='pastel')
plt.title('Frecuencia por Categoría')
plt.xlabel('Categoría')
plt.ylabel('Frecuencia')
plt.show()

# Crea visualizaciones univariantes con Seaborn
# Utiliza Seaborn para crear al menos un diagrama de caja (boxplot) 
# o un gráfico de violín (violinplot) para visualizar la distribución de una variable 
# numérica, posiblemente agrupada por una variable categórica.
sns.boxplot(x="Category", y="Sales", data=dataset)
plt.show()
sns.violinplot(x="Category", y="Profit", data=dataset)
plt.show()

#Implementa gráficos bivariantes con Matplotlib
#Crea un gráfico de dispersión o de líneas con Matplotlib para mostrar la relación 
# entre dos variables numéricas (por ejemplo, Ventas vs. Beneficios) o la evolución temporal 
# de una variable.

plt.scatter(dataset['Sales'], dataset['Profit'],  color='blue', s=80, edgecolor='black')
plt.title("Relación entre Ventas y Beneficios")
plt.xlabel("Ventas")
plt.ylabel("Beneficios")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

#Implementa gráficos bivariantes con Seaborn
#Utiliza Seaborn para crear un gráfico bivariante como un gráfico de barras agrupadas, 
# un gráfico de dispersión con regresión (regplot) o un gráfico de líneas mejorado.
sns.regplot(x="Sales", y="Profit", data=dataset, scatter_kws={"s": 60, "color": "blue"}, line_kws={"color": "red"})
plt.title("Relación entre Ventas y Beneficios con Regresión")
plt.xlabel("Ventas")
plt.ylabel("Beneficios")
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()

#Crea una visualización multivariante con Seaborn
#Implementa un heatmap de correlación o un pairplot para visualizar las relaciones entre múltiples variables numéricas del dataset.
numeric_dataset = dataset.select_dtypes(include=['int64', 'float64'])
correlation_matrix = numeric_dataset.corr()
plt.figure(figsize=(10,6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Matriz de Correlación')
plt.show()

#Organiza visualizaciones en subplots
#Crea una figura con múltiples subplots que muestre al menos 4 visualizaciones diferentes, organizadas de manera coherente 
# y con un título general.
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Visualizaciones del Dataset Superstore", fontsize=16, fontweight='bold')

axes[0, 0].hist(dataset['Sales'], bins=20, color='skyblue', edgecolor='black')
axes[0, 0].set_title("Distribución de Ventas")
axes[0, 0].set_xlabel("Ventas")
axes[0, 0].set_ylabel("Frecuencia")

sns.countplot(ax=axes[0, 1], x="Category", data=dataset, palette='pastel')
axes[0, 1].set_title("Frecuencia por Categoría")
axes[0, 1].set_xlabel("Categoría")
axes[0, 1].set_ylabel("Frecuencia")

axes[1, 0].scatter(dataset['Sales'], dataset['Profit'], color='green', edgecolors='black')
axes[1, 0].set_title("Relación Ventas vs Beneficios")
axes[1, 0].set_xlabel("Ventas")
axes[1, 0].set_ylabel("Beneficios")
axes[1, 0].grid(True, linestyle='--', alpha=0.5)

sns.boxplot(ax=axes[1, 1], x="Category", y="Sales", data=dataset, palette="Set2")
axes[1, 1].set_title("Ventas por Categoría")
axes[1, 1].set_xlabel("Categoría")
axes[1, 1].set_ylabel("Ventas")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

#Personaliza las visualizaciones
#Mejora la apariencia de tus gráficos añadiendo títulos descriptivos, etiquetas de ejes claras, leyendas cuando sea necesario 
# y utilizando paletas de colores apropiadas.
plt.figure(figsize=(8,5))
plt.hist(dataset['Sales'], bins=25, color='#4DA1FF', edgecolor='black', alpha=0.8)
plt.title("Distribución de Ventas en Superstore", fontsize=14, fontweight='bold')
plt.xlabel("Valor de Ventas", fontsize=12)
plt.ylabel("Frecuencia", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.4)
plt.show()

plt.savefig("histograma_ventas_superstore.png", dpi=300, bbox_inches="tight")