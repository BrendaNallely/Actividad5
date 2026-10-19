# Actividad5 — Marketing bancario
Repositorio: https://github.com/BrendaNallely/Actividad5

Predigo contratación de depósitos mediante regresión logística y Random Forest.

## Datos y atribución
Kaggle: https://www.kaggle.com/datasets/sahistapatel96/bankadditionalfullcsv
Fuente original: https://doi.org/10.24432/C5K306 (Moro, Rita y Cortez, 2014).
Licencia original UCI: CC BY 4.0. Datos históricos de campañas de un banco portugués.
Versión bank-additional-full: 41,188 filas y 21 columnas originales.
SHA256: 233e260d5d1d506a2c10381da5b8c2f75f2c08d1b373ca7b825e39ebe1bd30df. Fuente efectiva registrada en resultados/manifest_datos.json.

## Estructura
- datos/datos_ini/: CSV original sin modificar.
- datos/datos_limp/: datos limpios antes de imputación/escalado.
- fuentes/entrena.ipynb: cuaderno Colab.
- fuentes/datos_prep.py: función de limpieza.
- fuentes/train.py: entrenamiento, evaluación y registro MLflow.
- resultados/: diccionario, particiones, búsquedas, métricas y versiones.
- evidencias/: PNG comparativos, capturas reales y verificación MLflow.
- mlflow.db, mlartifacts/, modelos/: incluidos en respaldo ZIP; excluidos de Git.

## Reproducción
1. Abrir fuentes/entrena.ipynb en una sesión nueva de Colab con CPU.
2. Ejecutar en orden. Si cambia un paquete ya cargado, reiniciar una vez.
3. El ajuste completo realiza 110 fits de validación y dos reajustes.
4. Activar ABRIR_PANEL para capturas; después SUBIR_CAPTURAS.
5. Guardar el cuaderno ejecutado y generar el ZIP antes de cerrar Colab.

Alternativa local (Python 3.11–3.13):
    pip install -r requirements.txt
    python fuentes/train.py --root .
Prueba rápida funcional: añadir --smoke y usar un --root separado; no entregar sus métricas.
El script descarga Kaggle si falta el original y usa UCI como respaldo.

## Decisiones
Excluyo duration, conservo unknown y filas idénticas sin ID.
Recodifico pdays=999 junto con una bandera de ausencia de contacto.
Imputación y One-Hot dentro de cada fold; escalado solo en logística.
Prueba: último 20% por orden original. Cinco folds estratificados dentro del 80% inicial.
La validación interna mezcla periodos; no equivale a una evaluación temporal de producción.
Selección: F1 de validación, umbral 0.5. No reajusto según prueba.
Bootstrap pareado de F1 y McNemar de errores; suponen independencia entre filas.

## Resultado de esta ejecución
Seleccionado por validación: Random_Forest.
| modelo              |   accuracy |   precision |   recall |       f1 |   roc_auc |
|:--------------------|-----------:|------------:|---------:|---------:|----------:|
| Regresion_logistica |   0.489196 |    0.359026 | 0.83622  | 0.502365 |   0.69405 |
| Random_Forest       |   0.710852 |    0.619697 | 0.161024 | 0.255625 |   0.71814 |

## MLflow y restauración
El notebook verifica parámetros, métricas, artefactos y recarga de modelos.
Para restaurar el ZIP de Colab, conservar /content/Actividad5: los URI de artefactos son absolutos.
Desde un script local, mantener la misma ruta usada al entrenar, o regenerar el almacén.
    mlflow ui --backend-store-uri sqlite:////content/Actividad5/mlflow.db --port 5000
Las gráficas y tablas de Git permiten revisar resultados sin el almacén completo.

## Limitaciones y entrega
Datos antiguos, sin ID ni fecha exacta; no prueban causalidad ni desempeño en México actual.
El reporte PDF/Word se prepara por separado. Las capturas deben ser reales.
No se incluyen tokens, contraseñas ni archivos de autenticación en Git.
