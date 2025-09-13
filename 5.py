import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import altair as alt
import altair as alt
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
import umap
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target
import plotly.express as px
import io
import warnings

# Ignorar advertencias de KMeans (se producen con el uso de 'n_init="auto"')
warnings.filterwarnings('ignore', category=UserWarning)

st.set_page_config(
    page_title="Herramienta Interactiva de EDA y DR",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----- Utilidades generales -----
@st.cache_data(show_spinner="Cargando datos...")
def load_data(file=None):
    """Carga el archivo Excel desde una URL de GitHub o un archivo local."""
    GITHUB_XLSX_URL = "https://github.com/witman92/Examen_-Reducci-n-de-Dimensionalidad-y-Aplicaci-n-Interactiva-con-Streamlit/raw/274830aaee4edd806be40a74601f4a8218a102b7/Abulon.xlsx"
    try:
        if file:
            df = pd.read_excel(file)
        else:
            df = pd.read_excel(GITHUB_XLSX_URL)
        return df
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        return None


def get_numeric_cols(df):
    """Devuelve las columnas numéricas del DataFrame."""
    return df.select_dtypes(include=np.number).columns.tolist()


def get_cat_cols(df):
    """Devuelve las columnas categóricas del DataFrame."""
    return df.select_dtypes(include='object').columns.tolist()

# ----- Funciones de Visualización y Exploración -----
def plot_distribution(df, variable, selected_col, bins):
    """Genera y muestra un histograma y un boxplot/violinplot."""
    hist = alt.Chart(df).mark_bar().encode(
        x=alt.X(f"{variable}:Q", bin=alt.Bin(maxbins=bins), title=variable),
        y=alt.Y("count()", title="Frecuencia"),
        color=alt.Color(f"{selected_col}:N", title=selected_col) if selected_col else alt.value("steelblue"),
        tooltip=[variable, "count()"]
    ).properties(height=300)
    st.altair_chart(hist, use_container_width=True)

    if selected_col:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.violinplot(data=df, x=selected_col, y=variable, inner="quartile", ax=ax)
        ax.set_title(f"Distribución de {variable} por {selected_col}")
        st.pyplot(fig, clear_figure=True)


def plot_corr_matrix(df, num_cols):
    """Genera y muestra una matriz de correlación."""
    if len(num_cols) < 2:
        st.info("Se requieren al menos 2 columnas numéricas para mostrar la correlación.")
        return
    
    corr = df[num_cols].corr().round(2)
    corr_reset = corr.reset_index().rename(columns={'index': 'variable1'}).melt(
        id_vars="variable1", var_name="variable2", value_name="corr"
    )
    
    heat = alt.Chart(corr_reset).mark_rect().encode(
        x=alt.X("variable1:N", title=""),
        y=alt.Y("variable2:N", title=""),
        color=alt.Color("corr:Q", scale=alt.Scale(scheme="blueorange"), title="Correlación"),
        tooltip=["variable1", "variable2", "corr"]
    ).properties(height=400, width=400)
    
    st.altair_chart(heat, use_container_width=True)
    st.markdown("#### Pares de variables con mayor correlación")
    st.dataframe(
        corr_reset.assign(abs_corr=lambda d: d["corr"].abs()).sort_values(
            "abs_corr", ascending=False
        ).head(10).drop("abs_corr", axis=1), 
        use_container_width=True
    )


def exploracion_datos(df):
    """Interfaz para la exploración de datos."""
    st.title("Exploración Interactiva de Datos 📊")
    
    # --- Reporte de NA ---
    st.subheader("Reporte de valores faltantes (NA)")
    na_pct = (df.isnull().sum() / len(df) * 100).round(2)
    na_report = pd.DataFrame({"columna": df.columns, "pct_na": na_pct.values})
    na_report = na_report.sort_values("pct_na", ascending=False)
    st.dataframe(na_report[na_report["pct_na"]>0].reset_index(drop=True), use_container_width=True)
    if na_report["pct_na"].max() == 0:
        st.write("No se detectaron valores faltantes.")
    else:
        st.write("Indica la estrategia de imputación que vas a usar y por qué (breve):")
        impute_justif = st.text_area("Justificación de la estrategia de imputación", value="Breve justificación...")

    # Filtrado de datos
    cols_tres_valores = [col for col in df.columns if df[col].nunique() == 3]
    selected_col = st.sidebar.selectbox("Columna con 3 valores para filtrar", cols_tres_valores) if cols_tres_valores else None
    df_filtered = df.copy()
    
    if selected_col:
        opciones = sorted(df[selected_col].dropna().unique())
        seleccionados = st.sidebar.multiselect(f"Filtrar '{selected_col}'", opciones, default=opciones)
        if seleccionados:
            df_filtered = df[df[selected_col].isin(seleccionados)]
    
    st.info(f"Registros totales: {len(df)} | Registros filtrados: {len(df_filtered)}")
    st.dataframe(df_filtered.head(), use_container_width=True)
    
    num_cols = get_numeric_cols(df_filtered)
    cat_cols = get_cat_cols(df_filtered)

    with st.expander("Métricas y Estadísticas", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Columnas Numéricas", len(num_cols))
            if num_cols:
                desc = df_filtered[num_cols].describe().T
                st.dataframe(desc, use_container_width=True)
        with col2:
            st.metric("Columnas Categóricas", len(cat_cols))
            if cat_cols:
                st.dataframe(pd.DataFrame(df_filtered[cat_cols].describe().T), use_container_width=True)

    st.subheader("Visualización de Datos")
    if not num_cols:
        st.warning("No hay columnas numéricas para visualizar. Activa la codificación de categóricas o carga otro dataset.")
    else:
        variable = st.selectbox("Selecciona la variable numérica para analizar", num_cols)
        bins = st.slider("Número de Bins para Histograma", 5, 50, 20)
        plot_distribution(df_filtered, variable, selected_col, bins)

        st.subheader("Matriz de Correlación")
        plot_corr_matrix(df_filtered, num_cols)

    st.divider()
    csv = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar CSV Filtrado", data=csv, file_name="datos_filtrados.csv", mime="text/csv")

# ----- Funciones de Reducción de Dimensionalidad -----
@st.cache_data(show_spinner="Preprocesando datos...")
def preprocess_data(df, impute_strategy, scale_strategy, encode_cats, target_col=None, impute_justif=None, scale_justif=None):
    """Preprocesa los datos, lidiando con faltantes, escalado y codificación."""
    df_clean = df.copy()
    
    # Separar target si existe
    y = None
    if target_col and target_col in df_clean.columns:
        y = df_clean[target_col].copy()
        df_clean = df_clean.drop(columns=[target_col])
    
    numeric_cols = get_numeric_cols(df_clean)
    cat_cols = get_cat_cols(df_clean)

    # Imputación de valores faltantes en columnas numéricas
    if numeric_cols and df_clean[numeric_cols].isnull().values.any():
        try:
            imputer = KNNImputer() if impute_strategy == "knn" else SimpleImputer(strategy=impute_strategy)
            df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
        except Exception as e:
            st.error(f"Error al imputar los datos: {e}")
            return None, None

    # Codificación de columnas categóricas
    if encode_cats and cat_cols:
        for col in cat_cols:
            df_clean[col] = LabelEncoder().fit_transform(df_clean[col].astype(str))
    
    # Asegurarse de que todas las columnas son numéricas antes de escalar
    X = df_clean.select_dtypes(include=[np.number])
    if X.empty:
        st.error("No hay columnas numéricas para escalar. Asegúrate de que los datos no estén vacíos o activa la codificación de categóricas si es necesario.")
        return None, None
    
    # Escalado
    try:
        scaler = StandardScaler() if scale_strategy == "StandardScaler" else MinMaxScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    except Exception as e:
        st.error(f"Error al escalar los datos: {e}")
        return None, None

    return X_scaled, y

@st.cache_data
def get_embedding(method, X_train, X_test, y_train=None, params_tuple=None):
    """Aplica el método de reducción de dimensionalidad. params_tuple es (('param',value),...) para caching estable."""
    params = dict(params_tuple) if params_tuple else {}
    try:
        if method == "PCA":
            model = PCA(n_components=2)
            embedding_train = model.fit_transform(X_train)
            embedding_test = model.transform(X_test) if X_test is not None else None

            # Scree plot con varianza y varianza acumulada
            evr = model.explained_variance_ratio_
            df_evr = pd.DataFrame({
                "component": [f"PC{i+1}" for i in range(len(evr))],
                "variance_ratio": evr,
                "cumulative": np.cumsum(evr)
            })
            fig_scree = px.bar(df_evr, x="component", y="variance_ratio", hover_data=["variance_ratio","cumulative"],
                               title="Scree plot: Varianza explicada por componente")
            fig_scree.add_scatter(x=df_evr["component"], y=df_evr["cumulative"], mode="lines+markers", name="Cumulative")
            st.plotly_chart(fig_scree, use_container_width=True)

        elif method == "LDA":
            model = LDA(n_components=2)
            embedding_train = model.fit_transform(X_train, y_train)
            embedding_test = model.transform(X_test) if X_test is not None else None

        elif method == "t-SNE":
            st.warning("⚠️ t-SNE no transforma datos nuevos. Solo embedding de entrenamiento.")
            tsne_params = {k: v for k, v in params.items()}
            model = TSNE(n_components=2, **tsne_params)
            embedding_train = model.fit_transform(X_train)
            embedding_test = None

        elif method == "UMAP":
            umap_params = {k: v for k, v in params.items()}
            model = umap.UMAP(n_components=2, **umap_params)
            embedding_train = model.fit_transform(X_train)
            embedding_test = model.transform(X_test) if X_test is not None else None

        else:
            embedding_train, embedding_test = None, None

        return embedding_train, embedding_test
    except Exception as e:
        st.error(f"Error al aplicar el método {method}: {e}")
        return None, None


def plot_embedding(embedding_train, y_train, df_index=None):
    """Grafica la proyección 2D de los datos con tooltip id/label."""
    df_emb = pd.DataFrame(embedding_train, columns=["Dim1", "Dim2"])
    if df_index is not None:
        df_emb["id"] = df_index.tolist()
    if y_train is not None:
        df_emb["target"] = y_train.reset_index(drop=True).astype(str)

    hover_list = []
    if "id" in df_emb.columns:
        hover_list.append("id")
    if "target" in df_emb.columns:
        hover_list.append("target")

    fig = px.scatter(df_emb, x="Dim1", y="Dim2", color="target" if y_train is not None else None,
                     title="Proyección 2D del Embedding",
                     labels={"Dim1": "Componente 1", "Dim2": "Componente 2"},
                     hover_data=hover_list if hover_list else None)
    st.plotly_chart(fig, use_container_width=True)
    return df_emb, fig


def evaluate_embedding(embedding_train, y_train, embedding_test, y_test, method_name="Method"):
    """Muestra métricas de evaluación del embedding y devuelve resultados como dict."""
    results = {"method": method_name}

    if y_train is None:
        st.info("No hay variable target para evaluar. Se omiten las métricas supervisadas.")
        results["accuracy"] = np.nan
    else:
        # Métrica Supervisada: Accuracy con kNN
        if embedding_test is not None and y_test is not None:
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(embedding_train, y_train)
            y_pred = knn.predict(embedding_test)
            acc = accuracy_score(y_test, y_pred)
            st.metric(f"Accuracy kNN (test) - {method_name}", f"{acc:.4f}")
            results["accuracy"] = acc
        else:
            st.info("Accuracy no disponible (t-SNE o sin split).")
            results["accuracy"] = np.nan

    # Métrica No Supervisada: Silhouette Score
    try:
        k_clusters = len(np.unique(y_train)) if y_train is not None else 3
        kmeans = KMeans(n_clusters=max(2, k_clusters), n_init="auto", random_state=42)
        clusters = kmeans.fit_predict(embedding_train)
        sil_score = silhouette_score(embedding_train, clusters)
        st.metric(f"Silhouette Score (train) - {method_name}", f"{sil_score:.4f}")
        results["silhouette"] = sil_score
    except Exception as e:
        st.warning(f"Silhouette no pudo calcularse: {e}")
        results["silhouette"] = np.nan

    # ARI / NMI if target exists
    if y_train is not None:
        try:
            ari = adjusted_rand_score(y_train, clusters)
            nmi = normalized_mutual_info_score(y_train, clusters)
            st.write(f"ARI: {ari:.4f} | NMI: {nmi:.4f}")
            results["ARI"] = ari
            results["NMI"] = nmi
        except Exception as e:
            st.warning(f"ARI/NMI no pudieron calcularse: {e}")
            results["ARI"] = np.nan
            results["NMI"] = np.nan

    return results

# ----- Interfaz Principal -----
def main():
    st.sidebar.header("Opciones de Carga")
    use_github_file = st.sidebar.checkbox("Usar archivo de muestra desde GitHub", value=True)
    uploaded_file = st.sidebar.file_uploader("O sube un archivo Excel", type=["xlsx"])
    
    df = load_data(uploaded_file if uploaded_file else None)
    
    if df is None:
        st.warning("📄 Por favor, selecciona un archivo o usa el de ejemplo.")
        st.stop()
    
    st.sidebar.divider()
    app_mode = st.sidebar.radio("Selecciona la funcionalidad", ["Exploración de Datos", "Reducción de Dimensionalidad"])

    if app_mode == "Exploración de Datos":
        exploracion_datos(df)

    elif app_mode == "Reducción de Dimensionalidad":
        st.title("Reducción de Dimensionalidad Interactiva 📉")
        
        # Parámetros de preprocesamiento
        with st.sidebar.expander("Parámetros de Preprocesamiento", expanded=True):
            impute_strategy = st.selectbox("Estrategia de Imputación", ["median", "most_frequent", "knn"])
            scale_strategy = st.selectbox("Estrategia de Escalado", ["StandardScaler", "MinMaxScaler"])
            encode_cats = st.checkbox("Codificar variables categóricas")
            st.markdown("""
            **Explicación rápida sobre escalado:**
            - `StandardScaler`: centra a 0 y escala a desviación estándar 1. Preferible para PCA, métodos basados en distancia o cuando asumimos distribución cercana a normal.
            - `MinMaxScaler`: escala en el rango [0,1]. Útil para preservar rangos y para modelos sensibles a magnitud de features (ej. redes neuronales).
            """)
            scale_justif = st.text_area("Justificación del escalado (opcional)", value="Usé StandardScaler porque ...")
            impute_justif = st.text_area("Justificación de la imputación (opcional)", value="Usé median/knn/most_frequent porque ...")
        
        with st.sidebar.expander("Selección de Target", expanded=True):
            has_target = st.checkbox("¿Hay una variable objetivo (target)?", value=True)
            target_col = st.selectbox("Columna Target", df.columns) if has_target else None

        # Procesamiento y validación
        X_scaled, y = preprocess_data(df, impute_strategy, scale_strategy, encode_cats, target_col, impute_justif, scale_justif)
        if X_scaled is None: st.stop()
        
        if has_target and y is None:
            st.error("Error: La columna target seleccionada no está disponible o es inválida.")
            st.stop()
            
        test_size = st.sidebar.slider("Tamaño del set de test (%)", 10, 50, 30) / 100
        
        try:
            if has_target and y is not None:
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)
            else:
                X_train, X_test = train_test_split(X_scaled, test_size=test_size, random_state=42)
                y_train, y_test = None, None
        except Exception as e:
            st.error(f"Error al dividir los datos: {e}")
            st.stop()

        # Selección de método de reducción
        st.sidebar.header("Método de Reducción")
        method = st.sidebar.selectbox("Selecciona el método", ["PCA", "LDA", "t-SNE", "UMAP"])

        params = {}
        if method == "LDA" and (not has_target or (has_target and type_of_target(y) not in ['binary', 'multiclass'])):
            st.warning("LDA requiere una variable target categórica. Por favor, selecciona una o elige otro método.")
            st.stop()
        elif method == "t-SNE":
            with st.sidebar.expander("Parámetros de t-SNE"):
                params['perplexity'] = st.slider("Perplexity", 5, 50, 30, help="Controla el equilibrio entre la estructura local y global del embedding.")
                params['learning_rate'] = st.slider("Learning Rate", 10, 500, 200, help="Controla qué tan rápido se mueven los puntos.")
                params['random_state'] = 42
                st.markdown("""
                **Nota sobre t-SNE:** Sensible a `perplexity` (afecta la escala de la vecindad local) y `learning_rate` (estabilidad de la optimización). Cambios grandes pueden alterar drásticamente la forma final del embedding.
                """)
        elif method == "UMAP":
            with st.sidebar.expander("Parámetros de UMAP"):
                params['n_neighbors'] = st.slider("n_neighbors", 5, 50, 15, help="Controla el tamaño de la vecindad. Un valor bajo preserva la estructura local.")
                params['min_dist'] = st.slider("min_dist", 0.0, 0.99, 0.1, help="Controla la compactación de los agrupamientos.")
                params['random_state'] = 42
                st.markdown("""
                **Nota sobre UMAP:** `n_neighbors` afecta la escala de lo local/global; `min_dist` controla la densidad de agrupamientos y la separación entre clusters.
                """)

        compare_all = st.sidebar.checkbox("Comparar todos los métodos (ejecuta PCA, LDA (si aplica), t-SNE, UMAP)", value=False)

        # Convertir params a tuple ordenada para caching estable
        params_tuple = tuple(sorted(params.items())) if params else None

        # Generar y mostrar el embedding (o comparar todos)
        if compare_all:
            methods_to_run = ["PCA", "t-SNE", "UMAP"]
            if has_target and type_of_target(y) in ['binary', 'multiclass']:
                methods_to_run.append("LDA")

            method_results = []
            st.info("Comparando métodos. Esto puede tardar (especialmente t-SNE).")
            for m in methods_to_run:
                # reuse params for applicable methods, otherwise empty
                local_params = params if m in ["t-SNE", "UMAP"] and params else {}
                local_tuple = tuple(sorted(local_params.items())) if local_params else None
                emb_train, emb_test = get_embedding(m, X_train, X_test, y_train, params_tuple=local_tuple)
                if emb_train is None:
                    st.warning(f"No se pudo computar embedding para {m}.")
                    continue
                st.subheader(f"Método: {m}")
                df_vis, fig = plot_embedding(emb_train, y_train, df_index=X_train.index)
                res = evaluate_embedding(emb_train, y_train, emb_test, y_test, method_name=m)
                method_results.append(res)

                # Export botones para cada método
                csv_buffer = io.StringIO()
                df_vis.to_csv(csv_buffer, index=False)
                st.download_button(f"Descargar embedding CSV - {m}", csv_buffer.getvalue(), f"embedding_{m}.csv", "text/csv")
                try:
                    img_png = fig.to_image(format="png")
                    st.download_button(f"Descargar figura PNG - {m}", img_png, f"embedding_{m}.png", "image/png")
                    img_svg = fig.to_image(format="svg")
                    st.download_button(f"Descargar figura SVG - {m}", img_svg, f"embedding_{m}.svg", "image/svg+xml")
                except Exception as e:
                    st.warning(f"No se pudo exportar la figura para {m} (kaleido). Error: {e}")

            if method_results:
                df_comp = pd.DataFrame(method_results).set_index('method')
                st.subheader("Comparación de métodos")
                st.dataframe(df_comp, use_container_width=True)

        else:
            embedding_train, embedding_test = get_embedding(method, X_train, X_test, y_train, params_tuple=params_tuple)
            if embedding_train is None: st.stop()
            df_vis, fig = plot_embedding(embedding_train, y_train, df_index=X_train.index)
            
            with st.expander("Evaluación y Resultados", expanded=True):
                st.subheader("Evaluación del Embedding")
                res = evaluate_embedding(embedding_train, y_train, embedding_test, y_test, method_name=method)
                
                st.subheader("Exportación de Resultados")
                csv_buffer = io.StringIO()
                df_vis.to_csv(csv_buffer, index=False)
                st.download_button("Descargar embedding CSV", csv_buffer.getvalue(), "embedding.csv", "text/csv")
                try:
                    img_png = fig.to_image(format="png")
                    st.download_button("Descargar figura PNG", img_png, "embedding.png", "image/png")
                    img_svg = fig.to_image(format="svg")
                    st.download_button("Descargar figura SVG", img_svg, "embedding.svg", "image/svg+xml")
                except Exception as e:
                    st.warning(f"No se pudo exportar la imagen (kaleido). Error: {e}. Verifica que 'kaleido' esté instalado (pip install kaleido).")

                # Hallazgos automáticos
                st.subheader("Hallazgos clave (automático)")
                if res.get("silhouette") is not None and not np.isnan(res.get("silhouette")):
                    if res["silhouette"] > 0.5:
                        st.write("- El embedding muestra clusters bien definidos (silhouette > 0.5).")
                    elif res["silhouette"] > 0.25:
                        st.write("- Estructura moderada; puede haber subgrupos o solapamiento.")
                    else:
                        st.write("- Poca separación entre clusters; considerar otras técnicas o ajuste de parámetros.")
                if res.get("accuracy") is not None and not np.isnan(res.get("accuracy")):
                    st.write(f"- kNN accuracy sobre embedding: {res['accuracy']:.3f}.")
                else:
                    st.write("- No se pudo evaluar accuracy (posible uso de t-SNE o falta de split).")

                st.subheader("Limitaciones y recomendaciones")
                st.write("""
                - PCA: lineal, rápido, interpretable. Limitación: no captura relaciones no lineales. Recomendado como primer paso.
                - LDA: supervisado, favorece la separabilidad entre clases. Limitación: requiere target categórico y clases bien representadas.
                - t-SNE: excelente para visualización local, sensible a `perplexity` y `learning_rate`; no preserva distancias globales y no transforma nuevos datos (no es un embebido paramétrico).
                - UMAP: preserva mejor estructura global/local que t-SNE y permite transformar nuevos datos; parámetros `n_neighbors` controla la escala local/global y `min_dist` controla la densidad de los grupos.
                """)

if __name__ == "__main__":
    main()
