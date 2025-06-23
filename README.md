```
import joblib
import pandas as pd
```

1. Carga el modelo desde el archivo
```
filename = 'modelo_fumador_xxx.joblib'
modelo_cargado = joblib.load(filename)
print("¡Modelo cargado exitosamente!")
```

2. Prepara nuevos datos para predecir (deben tener la misma estructura que tu X_train original)
   Aquí creamos un ejemplo con dos perfiles de personas.
```
nuevos_datos = pd.DataFrame({
    'gender': ['M', 'F'],
    'age': [35, 50],
    'height(cm)': [175, 160],
    'waist(cm)': [3.71, 3.58],
    'relaxation': [3.13, 3.25],
    'fasting blood sugar': [3.92, 4.17],
    'triglyceride': [12.0, 3.2],
    'HDL': [1.88, 2.75],
    'LDL': [4.5, 5.0],
    'hemoglobin': [0.67, 0.51],
    'serum creatinine': [0.04, 0.01],
    'ALT': [2.1, 0.7],
    'Gtp': [3.5, 0.9],
    'dental caries': [1, 0],
    'tartar': ['Y', 'N']
})
```

3. Usa el modelo cargado para hacer predicciones
```
predicciones = modelo_cargado.predict(nuevos_datos)
probabilidades = modelo_cargado.predict_proba(nuevos_datos)
```
```
print("\n--- Resultados de la Predicción ---")
print(f"Predicciones (0=No Fumador, 1=Fumador): {predicciones}")
print(f"Probabilidades [No Fumador, Fumador]: \n{probabilidades}")
```
