"""Entrenamiento reproducible y registro MLflow de Actividad5.
Uso: python fuentes/train.py --root .
Prueba funcional (no entregar sus métricas): --smoke
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
parser=argparse.ArgumentParser()
parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
parser.add_argument('--smoke',action='store_true')
args=parser.parse_args()
proyecto=args.root.resolve()
MODO_PRUEBA_RAPIDA=args.smoke
from pathlib import Path
import os, sys, json, hashlib, shutil, time, platform, importlib.metadata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import binomtest
try:
    from IPython.display import display, Markdown
except ImportError:
    def display(obj): print(obj)
    def Markdown(text): return text
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# El script define proyecto antes de ejecutar estas instrucciones.
if 'proyecto' not in globals():
    proyecto = Path('/content/Actividad5') if Path('/content').exists() else Path.cwd()/'Actividad5'
proyecto = proyecto.resolve()
for sub in ['datos/datos_ini','datos/datos_limp','fuentes','resultados',
            'evidencias/mlflow','evidencias/comparacion_modelos','modelos']:
    (proyecto/sub).mkdir(parents=True, exist_ok=True)
RESULTADOS = proyecto/'resultados'
GRAFICAS = proyecto/'evidencias/comparacion_modelos'
SEED = 42
print('Directorio del proyecto:', proyecto)
# Guardo y muestro cada figura; el script usa backend Agg.
numero_figura=0
def mostrar_y_guardar():
    global numero_figura
    numero_figura += 1
    fig=plt.gcf()
    fig.savefig(GRAFICAS/f'{numero_figura:02d}_grafica.png',dpi=180,bbox_inches='tight')
    plt.show()
    plt.close(fig)


import kagglehub
import urllib.request, zipfile, io
archivo=proyecto/'datos/datos_ini/bank-additional-full.csv'
fuente='Archivo original conservado en el proyecto'
if not archivo.exists():
    try:
        path=kagglehub.dataset_download('sahistapatel96/bankadditionalfullcsv')
        candidatos=list(Path(path).rglob('bank-additional-full.csv'))
        if len(candidatos)!=1: raise FileNotFoundError('Versión completa no localizada')
        shutil.copy2(candidatos[0],archivo)
        fuente='Kaggle: sahistapatel96/bankadditionalfullcsv'
    except Exception as error:
        print('Kaggle no disponible; uso la fuente original UCI:',type(error).__name__)
        with urllib.request.urlopen('https://archive.ics.uci.edu/static/public/222/bank+marketing.zip',timeout=90) as r:
            z=zipfile.ZipFile(io.BytesIO(r.read()))
        directos=[x for x in z.namelist() if x.endswith('bank-additional-full.csv')]
        if directos:
            contenido=z.read(directos[0])
        else:
            interno=next(x for x in z.namelist() if x.endswith('bank-additional.zip'))
            zi=zipfile.ZipFile(io.BytesIO(z.read(interno)))
            contenido=zi.read(next(x for x in zi.namelist() if x.endswith('bank-additional-full.csv')))
        archivo.write_bytes(contenido)
        fuente='UCI original (respaldo de descarga)'
df=pd.read_csv(archivo,sep=';')
assert df.shape==(41188,21), f'Dataset inesperado: {df.shape}'
sha=hashlib.sha256(archivo.read_bytes()).hexdigest()
manifest={'sha256_original':sha,'fuente_efectiva':fuente,'filas':len(df),'columnas':len(df.columns)}
(RESULTADOS/'manifest_datos.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
print('Datos:',df.shape,'| SHA256:',sha)
display(df.head())


diccionario=pd.DataFrame([('age', 'Numérica', 'Edad en años'), ('job', 'Categórica', 'Ocupación'), ('marital', 'Categórica', 'Estado civil; divorced incluye viudez'), ('education', 'Categórica', 'Nivel educativo'), ('default', 'Categórica', 'Crédito en impago'), ('housing', 'Categórica', 'Préstamo hipotecario'), ('loan', 'Categórica', 'Préstamo personal'), ('contact', 'Categórica', 'Canal del contacto'), ('month', 'Categórica', 'Mes del contacto'), ('day_of_week', 'Categórica', 'Día del contacto'), ('duration', 'Numérica excluida', 'Duración de llamada en segundos'), ('campaign', 'Numérica', 'Contactos actuales; incluye el contacto actual'), ('pdays', 'Numérica', 'Días desde campaña previa; 999 indica sin contacto'), ('previous', 'Numérica', 'Contactos anteriores a la campaña actual'), ('poutcome', 'Categórica', 'Resultado de campaña previa'), ('emp.var.rate', 'Numérica', 'Variación trimestral de empleo'), ('cons.price.idx', 'Numérica', 'Índice mensual de precios'), ('cons.conf.idx', 'Numérica', 'Confianza mensual del consumidor'), ('euribor3m', 'Numérica', 'Euríbor a tres meses, diario'), ('nr.employed', 'Numérica', 'Indicador trimestral de número de empleados'), ('y', 'Objetivo', 'Contrató: yes=1, no=0')],columns=['variable','tipo','descripcion'])
diccionario.to_csv(RESULTADOS/'diccionario_datos.csv',index=False)
display(diccionario)


# Reviso la calidad de los datos del DataFrame.
diagnostico = pd.DataFrame({
    "tipo_original": df.dtypes.astype(str),
    "nulos": df.isna().sum(),
    "desconocidos": df.apply(
        lambda columna: columna.eq("unknown").sum()
    ),
    "valores_unicos": df.nunique()
})

display(diagnostico)

print("Registros:", len(df))
print("Filas idénticas:", df.duplicated().sum())
diagnostico.to_csv(RESULTADOS/'diagnostico_original.csv')
sys.path.insert(0,str(proyecto/'fuentes'))
from datos_prep import limpiar_datos
df_limpio=limpiar_datos(df)
df_limpio.to_csv(proyecto/'datos/datos_limp/bank_limpio.csv',index=False)
print('Datos limpios:',df_limpio.shape)
print('Filas idénticas conservadas:',df.duplicated().sum())


# CONFIGURACIÓN Y DATOS

AZUL_OSCURO = "#123B63"
AZUL_MEDIO = "#347FC4"
AZUL_CLARO = "#A9CDEB"
AZUL_SUAVE = "#EAF2FA"

sns.set_theme(
    style="white",
    rc={
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#D5DFE8",
        "axes.labelcolor": "#334155",
        "text.color": AZUL_OSCURO,
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "figure.dpi": 120
    }
)

# La prueba rápida usa una muestra y no sirve como resultado final.
if globals().get('MODO_PRUEBA_RAPIDA',False):
    df_limpio=df_limpio.sample(n=800,random_state=SEED).sort_index()
# Conservo el orden original y reservo el último 20% para prueba.
corte = int(len(df_limpio) * 0.80)

entrenamiento = df_limpio.iloc[:corte].copy()
prueba = df_limpio.iloc[corte:].copy()

X_train = entrenamiento.drop(columns="y")
y_train = entrenamiento["y"]

X_test = prueba.drop(columns="y")
y_test = prueba["y"]

assert set(entrenamiento["y"].unique()) == {0, 1}, (
    "La variable y debe estar codificada como 0 y 1."
)

print(f"Entrenamiento: {len(entrenamiento):,} registros")
print(f"Prueba reservada: {len(prueba):,} registros")


def estilo_limpio(ax, eje="y"):
    """Agrego una cuadrícula tenue sobre el eje indicado."""
    ax.set_axisbelow(True)
    ax.grid(axis=eje, color="#E8EEF4", linewidth=0.7)
    ax.tick_params(length=0)


# DISTRIBUCIÓN DE LA CONTRATACIÓN

conteos = (
    entrenamiento["y"]
    .value_counts()
    .reindex([0, 1], fill_value=0)
)

porcentajes = conteos / conteos.sum() * 100

fig, ax = plt.subplots(figsize=(8, 4.5))

barras = ax.bar(
    ["No contrató", "Sí contrató"],
    conteos.values,
    color=[AZUL_CLARO, AZUL_OSCURO],
    width=0.55
)

for barra, porcentaje in zip(barras, porcentajes):
    ax.annotate(
        f"{int(barra.get_height()):,}  ·  {porcentaje:.1f}%",
        xy=(
            barra.get_x() + barra.get_width() / 2,
            barra.get_height()
        ),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        fontsize=11,
        fontweight="bold"
    )

ax.set_title("Distribución de la contratación", loc="left", pad=20)
ax.set_ylabel("Observaciones")
ax.set_ylim(0, conteos.max() * 1.18)

estilo_limpio(ax)
plt.tight_layout()
mostrar_y_guardar()


# CONTRATACIÓN POR OCUPACIÓN

etiquetas_ocupacion = {
    "admin.": "Administración",
    "blue-collar": "Trabajo manual",
    "entrepreneur": "Emprendimiento",
    "housemaid": "Trabajo doméstico",
    "management": "Gerencia",
    "retired": "Jubilación",
    "self-employed": "Trabajo independiente",
    "services": "Servicios",
    "student": "Estudiante",
    "technician": "Personal técnico",
    "unemployed": "Desempleo",
    "unknown": "Desconocida"
}

por_ocupacion = (
    entrenamiento.groupby("job")["y"]
    .agg(tasa="mean", registros="size")
    .sort_values("tasa")
)

etiquetas = [
    etiquetas_ocupacion.get(ocupacion, ocupacion)
    for ocupacion in por_ocupacion.index
]

fig, ax = plt.subplots(figsize=(10, 6))

barras = ax.barh(
    etiquetas,
    por_ocupacion["tasa"],
    color=AZUL_MEDIO,
    height=0.65
)

for barra, (_, fila) in zip(barras, por_ocupacion.iterrows()):
    ax.annotate(
        f'{fila["tasa"]:.1%}  ·  n={int(fila["registros"]):,}',
        xy=(
            barra.get_width(),
            barra.get_y() + barra.get_height() / 2
        ),
        xytext=(7, 0),
        textcoords="offset points",
        va="center",
        fontsize=9
    )

ax.set_title(
    "Tasa de contratación por ocupación",
    loc="left",
    pad=20
)
ax.set_xlabel("Porcentaje de contratación")
ax.xaxis.set_major_formatter(PercentFormatter(1))
ax.set_xlim(0, max(por_ocupacion["tasa"].max() * 1.55, 0.10))

estilo_limpio(ax, eje="x")
plt.tight_layout()
mostrar_y_guardar()



# DISTRIBUCIÓN DE EDAD

fig, ax = plt.subplots(figsize=(9, 4.5))

sns.histplot(
    data=entrenamiento,
    x="age",
    bins=25,
    color=AZUL_MEDIO,
    edgecolor="white",
    alpha=0.9,
    ax=ax
)

mediana_edad = entrenamiento["age"].median()

ax.axvline(
    mediana_edad,
    color=AZUL_OSCURO,
    linestyle="--",
    linewidth=1.8,
    label=f"Mediana: {mediana_edad:.0f} años"
)

ax.set_title("Distribución de edad", loc="left", pad=20)
ax.set_xlabel("Edad en años")
ax.set_ylabel("Observaciones")
ax.legend(frameon=False)

estilo_limpio(ax)
plt.tight_layout()
mostrar_y_guardar()



# VALORES EXTREMOS


fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

variables = [
    ("age", "Edad", "Años"),
    ("campaign", "Contactos de campaña actual", "Contactos"),
    ("previous", "Contactos de campañas anteriores", "Contactos")
]

for ax, (columna, titulo, etiqueta) in zip(axes, variables):
    sns.boxplot(
        data=entrenamiento,
        y=columna,
        color=AZUL_CLARO,
        width=0.4,
        linewidth=1.2,
        flierprops={
            "marker": "o",
            "markersize": 3,
            "markerfacecolor": AZUL_MEDIO,
            "markeredgecolor": AZUL_MEDIO,
            "alpha": 0.35
        },
        medianprops={
            "color": AZUL_OSCURO,
            "linewidth": 2
        },
        ax=ax
    )

    ax.set_title(titulo, fontsize=12, pad=15)
    ax.set_ylabel(etiqueta)
    ax.set_xlabel("")
    estilo_limpio(ax)

fig.suptitle(
    "Revisión de valores extremos",
    fontsize=15,
    fontweight="bold",
    x=0.02,
    ha="left"
)

plt.tight_layout(rect=[0, 0, 1, 0.92])
mostrar_y_guardar()

print(
    "Los puntos fuera de los bigotes son candidatos a revisión; "
    "no se eliminan automáticamente."
)


# INFORMACIÓN DESCONOCIDA

categoricas = entrenamiento.select_dtypes(
    include=["object", "string"]
).columns

desconocidos = (
    entrenamiento[categoricas]
    .eq("unknown")
    .mean()
    .sort_values()
)

desconocidos = desconocidos[desconocidos > 0]

if not desconocidos.empty:
    fig, ax = plt.subplots(figsize=(9, 4.5))

    barras = ax.barh(
        desconocidos.index,
        desconocidos.values,
        color=AZUL_MEDIO,
        height=0.6
    )

    for barra, porcentaje in zip(barras, desconocidos.values):
        ax.annotate(
            f"{porcentaje:.1%}",
            xy=(
                barra.get_width(),
                barra.get_y() + barra.get_height() / 2
            ),
            xytext=(7, 0),
            textcoords="offset points",
            va="center"
        )

    ax.set_title("Información desconocida por variable", loc="left", pad=20)
    ax.set_xlabel("Registros con categoría unknown")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlim(0, desconocidos.max() * 1.25)

    estilo_limpio(ax, eje="x")
    plt.tight_layout()
    mostrar_y_guardar()
else:
    print("No se encontraron categorías unknown en entrenamiento.")



# CORRELACIONES NUMÉRICAS


correlaciones = (
    entrenamiento
    .select_dtypes(include="number")
    .corr()
)

mascara = np.triu(
    np.ones_like(correlaciones, dtype=bool),
    k=1
)

mapa_azules = LinearSegmentedColormap.from_list(
    "escala_azul",
    [AZUL_SUAVE, AZUL_CLARO, AZUL_OSCURO]
)

fig, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(
    correlaciones,
    mask=mascara,
    cmap=mapa_azules,
    vmin=-1,
    vmax=1,
    annot=True,
    fmt=".2f",
    annot_kws={"size": 9},
    linewidths=0.7,
    linecolor="white",
    square=True,
    cbar_kws={
        "label": "Correlación: −1 a +1",
        "shrink": 0.75
    },
    ax=ax
)

ax.set_title(
    "Correlaciones entre variables numéricas",
    loc="left",
    pad=20
)

plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
mostrar_y_guardar()

print(
    "En la matriz, los tonos claros representan correlaciones negativas "
    "y los oscuros positivas. Los números indican su magnitud."
)
pd.DataFrame({'fila_original':df_limpio.index,'particion':
    ['entrenamiento']*corte+['prueba']*(len(df_limpio)-corte)}).to_csv(RESULTADOS/'particiones.csv',index=False)
assert y_train.nunique()==y_test.nunique()==2


# Identifico los tipos de variables únicamente en entrenamiento.
columnas_numericas = X_train.select_dtypes(
    include="number"
).columns.tolist()

columnas_categoricas = X_train.select_dtypes(
    include=["object", "string"]
).columns.tolist()


def crear_preprocesador(escalar=False):
    # Preparo las variables numéricas.
    pasos_numericos = [
        ("imputacion", SimpleImputer(strategy="median"))
    ]

    if escalar:
        pasos_numericos.append(
            ("escalado", StandardScaler())
        )

    # Preparo las variables categóricas.
    pasos_categoricos = [
        (
            "imputacion",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "codificacion",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]

    return ColumnTransformer([
        (
            "numericas",
            Pipeline(pasos_numericos),
            columnas_numericas
        ),
        (
            "categoricas",
            Pipeline(pasos_categoricos),
            columnas_categoricas
        )
    ])


# Modelo 1: regresión logística con escalado numérico.
modelo_logistico = Pipeline([
    ("preprocesamiento", crear_preprocesador(escalar=True)),
    (
        "modelo",
        LogisticRegression(
            max_iter=2000,
            random_state=42
        )
    )
])

# Modelo 2: Random Forest sin escalado numérico.
modelo_random_forest = Pipeline([
    ("preprocesamiento", crear_preprocesador(escalar=False)),
    (
        "modelo",
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=1
        )
    )
])

# Conservo los modelos para el ajuste posterior.
modelos = {
    "Regresion_logistica": modelo_logistico,
    "Random_Forest": modelo_random_forest
}

for nombre, pipeline in modelos.items():
    print(f"\n{nombre}")
    display(pipeline)

# Defino las configuraciones que voy a comparar.
hiperparametros = {
    "Regresion_logistica": {
        "modelo__C": [0.1, 1.0, 10.0],
        "modelo__class_weight": [None, "balanced"]
    },

    "Random_Forest": {
        "modelo__n_estimators": [100, 200],
        "modelo__max_depth": [8, None],
        "modelo__min_samples_leaf": [2, 10],
        "modelo__class_weight": [None, "balanced"]
    }
}

# Creo los mismos cinco folds para ambos modelos.
validacion = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

folds = list(validacion.split(X_train, y_train))

# Registro varias métricas y selecciono por F1.
metricas_validacion = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}


if globals().get('MODO_PRUEBA_RAPIDA',False):
    hiperparametros={
        'Regresion_logistica':{'modelo__C':[1.0]},
        'Random_Forest':{'modelo__n_estimators':[5],'modelo__max_depth':[3]}}
    folds=list(StratifiedKFold(n_splits=2,shuffle=True,random_state=SEED).split(X_train,y_train))

mlflow.set_tracking_uri('sqlite:///'+str(proyecto/'mlflow.db'))
cliente_mlflow=mlflow.tracking.MlflowClient()
nombre_experimento=('Actividad5_PRUEBA_RAPIDA' if globals().get('MODO_PRUEBA_RAPIDA',False)
                    else 'Actividad5_Marketing_Bancario')
experimento=cliente_mlflow.get_experiment_by_name(nombre_experimento)
id_experimento=(experimento.experiment_id if experimento else
    cliente_mlflow.create_experiment(nombre_experimento,artifact_location=(proyecto/'mlartifacts').as_uri()))
mlflow.set_experiment(experiment_id=id_experimento)

# Conservo versiones, datos y partición para trazabilidad.
paquetes=['numpy','pandas','scikit-learn','scipy','matplotlib','seaborn','mlflow','pyarrow','kagglehub','joblib','sqlalchemy']
versiones={p:importlib.metadata.version(p) for p in paquetes}
(RESULTADOS/'entorno.json').write_text(json.dumps({'python':sys.version,'paquetes':versiones},indent=2))
(proyecto/'requirements.txt').write_text('\n'.join(f'{p}=={v}' for p,v in versiones.items())+'\n')


busquedas={}; mejores_modelos={}; resumen=[]; ids_ejecuciones={}
# El checkpoint evita repetir una búsqueda completada si ejecuto la celda otra vez.
# Solo cargo checkpoints creados por este proyecto; joblib no es seguro con archivos ajenos.
for nombre,pipeline in modelos.items():
    firma=hashlib.sha256(json.dumps({'sha':sha,'version_codigo':'1.1.0',
        'params':hiperparametros[nombre],'versiones':versiones,
        'n_train':len(X_train),'indice_train':X_train.index.tolist(),
        'folds':len(folds),'seed':SEED},sort_keys=True).encode()).hexdigest()
    checkpoint=proyecto/'modelos'/f'{nombre}_{firma[:12]}.joblib'
    with mlflow.start_run(run_name=nombre) as run:
        ids_ejecuciones[nombre]=run.info.run_id
        mlflow.set_tags({'version_codigo':'1.1.0','sha256_original':sha,
                        'modo':'prueba_rapida' if globals().get('MODO_PRUEBA_RAPIDA',False) else 'completo'})
        mlflow.log_params({'seed':SEED,'folds':len(folds),'umbral':0.5,'seleccion':'f1',
            'particion':'80_20_orden_original','n_train':len(X_train),'n_test':len(X_test),
            'duration_excluida':True})
        if checkpoint.exists():
            busqueda=joblib.load(checkpoint)
            mlflow.set_tag('origen_ajuste','checkpoint_compatible')
            print('Recuperado:',nombre)
        else:
            print('Ajustando:',nombre,flush=True)
            busqueda=GridSearchCV(pipeline,hiperparametros[nombre],scoring=metricas_validacion,
                refit='f1',cv=folds,n_jobs=2,verbose=1,error_score='raise')
            inicio=time.perf_counter()
            busqueda.fit(X_train,y_train)
            mlflow.log_metric('segundos_ajuste',time.perf_counter()-inicio)
            mlflow.set_tag('origen_ajuste','entrenamiento_nuevo')
            joblib.dump(busqueda,checkpoint)
        busquedas[nombre]=busqueda
        mejores_modelos[nombre]=busqueda.best_estimator_
        indice=busqueda.best_index_
        fila={'modelo':nombre,'mejores_parametros':busqueda.best_params_}
        for m in metricas_validacion:
            valor=float(busqueda.cv_results_[f'mean_test_{m}'][indice])
            fila[f'cv_{m}']=valor
            mlflow.log_metric(f'cv_{m}',valor)
        fila['cv_f1_std']=float(busqueda.cv_results_['std_test_f1'][indice])
        mlflow.log_metric('cv_f1_std',fila['cv_f1_std'])
        resumen.append(fila)
        mlflow.log_params({k:str(v) for k,v in busqueda.best_params_.items()})
        tabla_cv=pd.DataFrame(busqueda.cv_results_)
        ruta_cv=RESULTADOS/f'gridsearch_{nombre}.csv'
        tabla_cv.to_csv(ruta_cv,index=False)
        mlflow.log_artifact(str(ruta_cv),'validacion')
        # Cada configuración aparece como ejecución hija en el panel.
        for i,row in tabla_cv.iterrows():
            with mlflow.start_run(run_name=f'config_{i:02d}',nested=True):
                mlflow.log_params({k:str(v) for k,v in row['params'].items()})
                mlflow.log_metrics({f'cv_{m}':float(row[f'mean_test_{m}']) for m in metricas_validacion})
        ejemplo=X_train.head(5)
        mlflow.sklearn.log_model(busqueda.best_estimator_,artifact_path='modelo',
            signature=infer_signature(ejemplo,busqueda.predict(ejemplo)),input_example=ejemplo)
        mlflow.log_artifact(str(RESULTADOS/'manifest_datos.json'),'datos')
        mlflow.log_artifact(str(RESULTADOS/'entorno.json'),'entorno')
resumen_ajuste=pd.DataFrame(resumen).set_index('modelo').sort_values('cv_f1',ascending=False)
modelo_seleccionado=resumen_ajuste.index[0]
resumen_ajuste.to_csv(RESULTADOS/'metricas_validacion.csv')
display(resumen_ajuste.drop(columns='mejores_parametros').round(4))
print('Seleccionado únicamente por validación:',modelo_seleccionado)



nombres = ["Regresion_logistica", "Random_Forest"]

if "mejores_modelos" not in globals():
    raise RuntimeError("Primero ejecuta el ajuste de hiperparámetros.")

for nombre in nombres:
    if nombre not in mejores_modelos:
        raise RuntimeError(f"Falta terminar el ajuste de {nombre}.")

predicciones = {}
filas_metricas = []

for nombre in nombres:
    modelo = mejores_modelos[nombre]

    # Obtengo la probabilidad de contratación.
    indice_positivo = list(modelo.classes_).index(1)
    probabilidades = modelo.predict_proba(X_test)[:, indice_positivo]

    # Mantengo el mismo umbral para ambos modelos.
    prediccion = (probabilidades >= 0.5).astype(int)
    predicciones[nombre] = prediccion

    filas_metricas.append({
        "modelo": nombre,
        "accuracy": accuracy_score(y_test, prediccion),
        "precision": precision_score(
            y_test, prediccion, zero_division=0
        ),
        "recall": recall_score(
            y_test, prediccion, zero_division=0
        ),
        "f1": f1_score(
            y_test, prediccion, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_test, probabilidades)
    })

tabla_metricas = (
    pd.DataFrame(filas_metricas)
    .set_index("modelo")
)

print("MÉTRICAS DE LOS MODELOS OPTIMIZADOS EN PRUEBA")
display(tabla_metricas.round(4))

print(
    "La clase positiva es contratación (y=1). "
    "MAE, RMSE y R² no aplican porque es clasificación."
)

# Gráfica comparativa en tonos azules.
fig, ax = plt.subplots(figsize=(11, 5))

tabla_metricas.T.plot.bar(
    ax=ax,
    color=["#123B63", "#79AFE0"],
    rot=0,
    width=0.7
)

ax.set_title("Desempeño de los modelos optimizados", loc="left")
ax.set_xlabel("")
ax.set_ylabel("Valor de la métrica")
ax.set_ylim(0, 1.15)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#E8EEF4")
ax.set_axisbelow(True)
ax.legend(frameon=False)

for contenedor in ax.containers:
    ax.bar_label(contenedor, fmt="%.3f", padding=3, fontsize=8)

plt.tight_layout()
mostrar_y_guardar()

# COMPARACIÓN ESTADÍSTICA DE RESULTADOS

y_real = np.asarray(y_test)
pred_lr = predicciones["Regresion_logistica"]
pred_rf = predicciones["Random_Forest"]

# Bootstrap estratificado pareado:
# remuestreo las mismas observaciones para ambos modelos,
# conservando el número de casos de cada clase.
rng = np.random.default_rng(42)

grupos = [
    np.flatnonzero(y_real == clase)
    for clase in [0, 1]
]

diferencias = []

for _ in range(1000):
    indices = np.concatenate([
        rng.choice(grupo, size=len(grupo), replace=True)
        for grupo in grupos
    ])

    diferencias.append(
        f1_score(
            y_real[indices], pred_rf[indices], zero_division=0
        )
        - f1_score(
            y_real[indices], pred_lr[indices], zero_division=0
        )
    )

limite_inferior, limite_superior = np.quantile(
    diferencias, [0.025, 0.975]
)

diferencia_f1 = (
    tabla_metricas.loc["Random_Forest", "f1"]
    - tabla_metricas.loc["Regresion_logistica", "f1"]
)

# McNemar exacto: comparo los errores discordantes.
solo_lr_acierta = int(
    ((pred_lr == y_real) & (pred_rf != y_real)).sum()
)

solo_rf_acierta = int(
    ((pred_lr != y_real) & (pred_rf == y_real)).sum()
)

discordantes = solo_lr_acierta + solo_rf_acierta

p_valor = (
    binomtest(
        solo_lr_acierta,
        n=discordantes,
        p=0.5,
        alternative="two-sided"
    ).pvalue
    if discordantes > 0
    else 1.0
)

comparacion_estadistica = pd.DataFrame({
    "resultado": {
        "Diferencia F1 (RF − logística)": diferencia_f1,
        "IC 95%: límite inferior": limite_inferior,
        "IC 95%: límite superior": limite_superior,
        "Solo logística acierta": solo_lr_acierta,
        "Solo Random Forest acierta": solo_rf_acierta,
        "McNemar: p-valor": p_valor
    }
})

print("\nCOMPARACIÓN ESTADÍSTICA")
display(comparacion_estadistica)

if limite_inferior > 0:
    print("F1: el intervalo favorece a Random Forest.")
elif limite_superior < 0:
    print("F1: el intervalo favorece a regresión logística.")
else:
    print(
        "F1: el intervalo incluye cero; no encuentro evidencia "
        "concluyente de una diferencia. Esto no demuestra equivalencia."
    )

if p_valor < 0.05:
    print(
        "McNemar: detecto diferencias en las tasas de error "
        "al nivel de significancia de 0.05."
    )
else:
    print(
        "McNemar: no detecto diferencias en las tasas de error "
        "al nivel de significancia de 0.05."
    )

# Visualizo la incertidumbre de la diferencia de F1.
fig, ax = plt.subplots(figsize=(9, 4))

ax.hist(
    diferencias,
    bins=35,
    color="#79AFE0",
    edgecolor="white"
)

ax.axvline(
    0, color="#64748B", linestyle="--", label="Sin diferencia"
)
ax.axvline(
    limite_inferior, color="#123B63", label="Límites del IC 95%"
)
ax.axvline(limite_superior, color="#123B63")

ax.set_title("Bootstrap pareado: diferencia de F1", loc="left")
ax.set_xlabel("F1 Random Forest − F1 regresión logística")
ax.set_ylabel("Remuestreos")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)

plt.tight_layout()
mostrar_y_guardar()
tabla_metricas.to_csv(RESULTADOS/'metricas_prueba.csv')
comparacion_estadistica.to_csv(RESULTADOS/'comparacion_estadistica.csv')
print('La inferencia supone filas independientes y no incorpora incertidumbre del entrenamiento.')
print('McNemar compara errores, no F1. No reajusto el modelo usando prueba.')


# Completo las ejecuciones principales con métricas de prueba y artefactos.
for nombre in nombres:
    with mlflow.start_run(run_id=ids_ejecuciones[nombre]):
        mlflow.log_metrics({'test_'+m:float(tabla_metricas.loc[nombre,m]) for m in tabla_metricas.columns})
        mlflow.set_tag('seleccionado_por_validacion',str(nombre==modelo_seleccionado))
        mlflow.log_artifacts(str(GRAFICAS),'graficas')
        for p in ['metricas_prueba.csv','comparacion_estadistica.csv','metricas_validacion.csv','diccionario_datos.csv']:
            mlflow.log_artifact(str(RESULTADOS/p),'resultados')
        mlflow.log_artifacts(str(proyecto/'fuentes'),'codigo')
# Verifico parámetros, métricas, artefactos y carga del modelo registrado.
filas=[]
for nombre,rid in ids_ejecuciones.items():
    r=cliente_mlflow.get_run(rid)
    assert r.info.status=='FINISHED' and r.data.params and 'test_f1' in r.data.metrics
    assert cliente_mlflow.list_artifacts(rid,'modelo')
    recuperado=mlflow.sklearn.load_model(f'runs:/{rid}/modelo')
    np.testing.assert_array_equal(recuperado.predict(X_test.head(5)),mejores_modelos[nombre].predict(X_test.head(5)))
    filas.append({'modelo':nombre,'run_id':rid,'estado':r.info.status,**r.data.metrics})
evidencia=pd.DataFrame(filas)
display(evidencia)
evidencia.to_csv(proyecto/'evidencias/mlflow/ejecuciones_verificadas.csv',index=False)
cliente_mlflow.search_runs([id_experimento])
(RESULTADOS/'run_ids.json').write_text(json.dumps(ids_ejecuciones,indent=2))
print('MLflow verificado: parámetros, métricas, artefactos y predicción del modelo recuperado.')
