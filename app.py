
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import chart_studio.plotly as py

from typing import Dict, Tuple, List
import json


def main():
    st.set_page_config(layout="wide", 
                       page_title="Grace", 
                       page_icon=":satellite:",)

    #plot_grace_picture()
    plot_web_clustering = False
    _, col_results, _ = st.columns([1, 4, 1])

    with col_results:
        st.title('Clustering GRACE time series :earth_americas: :satellite:')

        col1, col2 = st.columns([1, 6])
        col1.markdown('**Methodology:**')
        
        col2.write('''We are looking for geographical areas with similar trends in their time series.
        For this, we need to interpolate the missing data. After that, we decompose each serie into a trend and seasonality. After that, we use the trends for clusterize each coordinate.
                You can read more on our [medium article](https://medium.com/@felipe.salas.b/3a4890bd4428).''')

    _, col1, col2, _ = st.columns(4)
    labs = {"Center for Space Research at University of Texas (CSR)": "CSR",
            "Jet Propulsion Laboratory (JPL)": "JPL",
            "GeoforschungsZentrum Potsdam (GFZ)": "GFZ"
    }
    centro = col1.radio('Select a center', labs.keys())
    centro = labs[centro]
    num_cluster = col2.number_input('Select the number of clusters', 2, 10, 5)

    _, col_results, _ = st.columns([1, 4, 1])

    with col_results:

        with st.expander('1. How to choose the correct number of clusters'):
            st.subheader('1. Choosing the number of clusters')
            url_inertia = "https://scikit-learn.org/stable/modules/clustering.html#k-means"
            url_silhouette = "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html"
            url_davies_bouldin = "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html"
            st.write("""Although choosing the right number of clusters is a hard decision, there are several heuristics that help us in that task.
                    We have avaiable three metrics: [inertia](%s),  [silhouette score](%s) y [davies bouldin score](%s) which indicates how well clusterized was the data""" % (url_inertia, url_silhouette, url_davies_bouldin))
            
            plot_clustering_optimal_k(centro)

        tab_mapa, tab_caracterizacion, tab_numero_coordenadas  = st.tabs(['Map of each cluster', 'Cluster characterization', 'Number of coordinates of each cluster'])
        with tab_mapa:
            diccionario_colores = plot_resultados_clustering(num_cluster, centro)
        
        with tab_numero_coordenadas:
            plot_numero_coordenadas(centro, num_cluster, diccionario_colores)


        with tab_caracterizacion:
            st.subheader('Cluster interpretability')
            st.write('''We can characterize each cluster by looking at a sample of trends that belong to it. 
                    We can also look at the median of the sample to see the most representative trend of the cluster (the black curves)''')

            plot_caracterizacion_clusters_resumen(num_cluster, centro, diccionario_colores, 1000)
            
            plot_caracterizacion_clusters(num_cluster, centro, 1000)     
        




def plot_filogenia(
    dataf: pd.DataFrame, min_cluster: int, max_cluster: int, plot_plotly_web: bool = False
) -> go.Figure:
    """Grafica la filogenia de una base clusterizada con diferentes k."""
    labels, source, target, value = calculate_labels_source_target_value(
        dataf, min_cluster=min_cluster, max_cluster=max_cluster
    )

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color=px.colors.qualitative.Safe[0], width=0.5),
                    label=labels,
                    color=px.colors.qualitative.Safe[1],
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value,
                    color=px.colors.qualitative.Pastel2[2],
                    arrowlen=15
                ),
            )
        ]
    )
    fig.update_layout(
        title_text="Phylogeny of each cluster",
        font_size=10,
        width=1000,
    )
    if plot_plotly_web:
        py.plot(fig, filename="filogenia", auto_open = True)

    return fig


def calculate_labels_source_target_value(
    dataf: pd.DataFrame, min_cluster: int, max_cluster: int
) -> Tuple[List[str], List[int], List[int], List[int]]:
    """Calcula los labels de source, target y value para cada cluster.

    El dataframe de entrada tiene que tener las columnas
    de pertenencia a cada cluster.
    """
    aux_col = dataf.columns[0]

    labels, source, target, value = [], [], [], []
    k_accumulated = 0

    for k in range(min_cluster, max_cluster + 1):
        labels += [f"{k}clusters_{i+1}" for i in range(k)]
        if k < max_cluster:
            cols_to_groupby = [f"clusters_{k}", f"clusters_{k+1}"]
            _df = (
                dataf.groupby(cols_to_groupby)[[aux_col]]
                .count()
                .reset_index()
                .sort_values(by=cols_to_groupby)
            )

            source += [
                i + k_accumulated - 1 for i in _df[f"clusters_{k}"].tolist()
            ]

            k_accumulated += k
            target += [
                i + k_accumulated - 1 for i in _df[f"clusters_{k+1}"].tolist()
            ]
            value += _df[aux_col].tolist()

    return labels, source, target, value


def plot_clustering_optimal_k(centro: str, plot_plotly_web: bool = False) -> None:
    inertia, silhouette_scores, davies_bouldin_scores = load_scores(centro=centro)
    
    df_cluster = pd.read_parquet(f'data_for_app/data_cluster_trend_{centro}.parquet')
    tab1, tab2, tab3, tab4 = st.tabs(['Inertia', 'Silhouette', 'Davies Bouldin', 'Filogenia'])
    with tab1:
        fig_inertia = plot_elbow(inertia, figure_width=1000, plot_plotly_web=plot_plotly_web)
        st.plotly_chart(fig_inertia)
    with tab2:
        fig_silhouette = plot_silhouette(silhouette_scores, figure_width=1000, plot_plotly_web=plot_plotly_web)
        st.plotly_chart(fig_silhouette)
    with tab3:
        fig_davies_bouldin = plot_davies_bouldin(davies_bouldin_scores, figure_width=1000, plot_plotly_web=plot_plotly_web)
        st.plotly_chart(fig_davies_bouldin)
    with tab4:
        fig_filogenia = plot_filogenia(df_cluster, min_cluster=2, max_cluster=8, plot_plotly_web=plot_plotly_web)
        st.plotly_chart(fig_filogenia)


def plot_elbow(inertia: Dict[int, float], figure_width: int, plot_plotly_web: bool = False) -> go.Figure:
    """Graficamos el diagrama de elbow.

    Args:
        inertia:
        figure_width:

    Returns:
        Figura de elbow
    """
    fig = px.line(x=inertia.keys(), y=inertia.values())
    fig.update_layout(
        title="Elbows graph", width=figure_width, hovermode=False, xaxis_title='Number of cluster', yaxis_title='Inertia'
    )
    if plot_plotly_web:
        py.plot(fig, filename="inertia", auto_open = True)
    # fig.add_annotation(
    #     text="We are looking for the \"elbow\" point for deciding the optimal k",
    #     xref="paper",
    #     yref="paper",
    #     x=0.8,
    #     y=0.8,
    #     showarrow=False,
    # )
    return fig


def plot_silhouette(
    silhouette_score: Dict[int, float], figure_width: int, plot_plotly_web: bool = False
) -> go.Figure:
    """Graficamos el diagrama de silhouette."""
    fig = px.line(x=silhouette_score.keys(), y=silhouette_score.values())
    fig.update_layout(
        title="Silhouette graph", width=figure_width, hovermode=False, xaxis_title='Number of cluster', yaxis_title='Silhouette score'
    )
    # fig.add_annotation(
    #     text="We are looking for highest values for deciding the optimal k",
    #     xref="paper",
    #     yref="paper",
    #     x=0.8,
    #     y=0.8,
    #     showarrow=False,
    # )
    if plot_plotly_web:
        py.plot(fig, filename="silhouette", auto_open = True)

    return fig


def plot_davies_bouldin(
    davies_bouldin_scores: Dict[int, float], figure_width: int, plot_plotly_web: bool = False
) -> go.Figure:
    """Graficamos el diagrama de davies_bouldin."""
    fig = px.line(
        x=davies_bouldin_scores.keys(), y=davies_bouldin_scores.values()
    )
    fig.update_layout(
        title="Davies Bouldin graph", width=figure_width, hovermode=False, xaxis_title='Number of cluster', yaxis_title='Davies Bouldin score'
    )
    # fig.add_annotation(
    #     text="Looking for the smallest values for deciding the optimal k",
    #     xref="paper",
    #     yref="paper",
    #     x=0.9,
    #     y=0.2,
    #     showarrow=False,
    # )
    if plot_plotly_web:
        py.plot(fig, filename="davies_bouldin", auto_open = True)
    return fig


def load_scores(centro):
    
    with open(f'./data_for_app/scores_{centro}.json', 'r') as f:
        scores = json.load(f)
    inertia = scores['inertia']
    silhouette_scores = scores['silhouette_scores']
    davies_bouldin_scores = scores['davies_bouldin_scores']
    return inertia, silhouette_scores, davies_bouldin_scores
    

def plot_caracterizacion_clusters_resumen(num_cluster, centro, diccionario_colores, figure_width: int = 800, plot_plotly_web: bool = False):
    '''Graficamos las trazas de cada cluster'''

    path_trends = f'./data_for_app/trends_per_coordinate_{centro}.parquet'
    path_clusters = f'./data_for_app/data_cluster_trend_{centro}.parquet'

    trends = pd.read_parquet(path_trends)
    clusters = pd.read_parquet(path_clusters)
    df = trends.merge(clusters, on=['lat', 'lon'])

    
    cols_date = [m for m in df.columns if '-' in str(m)]

    median_representative = {}
    for cluster in df[f'clusters_{int(num_cluster)}'].unique():

        data = df.query(f'clusters_{int(num_cluster)}=={cluster}').sample(200, replace=True)
        trends_individuales = [go.Scatter(x=cols_date, y=data.iloc[j], opacity=0.2, mode='lines',
                                        marker_color='#B7B7B7', showlegend=False, hoverinfo='skip') for j in range(data.shape[0])]
        true_median = np.median(data[cols_date].values, axis=0)
        
        median_representative[cluster] = data.iloc[[np.argmin(np.linalg.norm(data[cols_date] - true_median, axis=1))]][cols_date]

    fig = go.Figure([go.Scatter(x=cols_date, 
                                y=data.values[0], 
                                mode='lines', 
                                name=f'cluster {k}', 
                                hoverinfo='skip', 
                                line_color=diccionario_colores[f'cluster_{k}']) 
                                for k, data in median_representative.items()])
    fig.update_layout(title=f'Comparision between each cluster representant', width=figure_width)
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.2,
        xanchor="right",
        x=1
    ),
    yaxis_title='lwe thickness')
    if plot_plotly_web:
        fig.update_layout(title=f'Comparision between each cluster representant', width=1000, template='plotly_white')
        fig.write_image(f"representative_clusters.png")
        py.plot(fig, filename=f"representative_clusters", auto_open = True)

    st.plotly_chart(fig)


def plot_caracterizacion_clusters(num_cluster: int, centro: str, figure_width: int = 800, plot_plotly_web: bool = False):
    '''Graficamos las trazas de cada cluster'''

    path_trends = f'./data_for_app/trends_per_coordinate_{centro}.parquet'
    path_clusters = f'./data_for_app/data_cluster_trend_{centro}.parquet'

    trends = pd.read_parquet(path_trends)
    clusters = pd.read_parquet(path_clusters)
    df = trends.merge(clusters, on=['lat', 'lon'])

    cols_date = [m for m in df.columns if '-' in str(m)]

    median_representative = {}
    for cluster in df[f'clusters_{int(num_cluster)}'].unique():

        data = df.query(f'clusters_{int(num_cluster)}=={cluster}').sample(60, replace=True)
        trends_individuales = [go.Scatter(x=cols_date, y=data.iloc[j], opacity=0.2, mode='lines',
                                        marker_color='#B7B7B7', showlegend=False, hoverinfo='skip') for j in range(data.shape[0])]
        true_median = np.median(data[cols_date].values, axis=0)
        
        median_representative[cluster] = data.iloc[[np.argmin(np.linalg.norm(data[cols_date] - true_median, axis=1))]][cols_date]
        trends_agregados = [go.Scatter(x=cols_date, y=list(median_representative[cluster].values[0]), opacity=0.8, mode='lines',
                                        marker_color='black', showlegend=False, hoverinfo='skip')] 
        fig = go.Figure(data=trends_individuales+trends_agregados)
        fig.update_layout(title=f'Samples of trends from cluster {cluster}', width=figure_width, yaxis_title='lwe thickness')

        if plot_plotly_web: 
            fig.update_layout(title=f'Samples of trends from cluster {cluster} and their representative', width=1000, template = 'plotly_white')
            fig.write_image(f"samples_cluster_{cluster}.png")
            py.plot(fig, filename=f"samples_clusters_{cluster}", auto_open = True)

        st.plotly_chart(fig)

    
def plot_numero_coordenadas(centro: str, nclusters: int, diccionario_colores: Dict, plot_plotly_web: bool = False) -> None:
    
    n_clusters = int(nclusters)

    df_cluster = pd.read_parquet(f'data_for_app/data_cluster_trend_{centro}.parquet')
    df_toplot = df_cluster[['lat', 'lon', f'clusters_{n_clusters}']].copy()

    df = df_toplot.copy()[f'clusters_{n_clusters}'].value_counts().reset_index()
    df.rename(columns={'count': 'número de coordenadas', f'clusters_{n_clusters}': 'cluster'}, inplace=True)

    df['Percentage'] = (df['número de coordenadas'] / df['número de coordenadas'].sum()) * 100
    df['color'] = df['cluster'].apply(lambda x: diccionario_colores[f'cluster_{x}'])


    fig = go.Figure(data=[go.Bar(
        x=df['cluster'],
        y=df['número de coordenadas'],
        text=df['Percentage'].round(2).astype(str) + '%',
        textposition='auto',
        marker_color=df['color']
    )])
    fig.update_layout(
        title='Number of coordinates associated to each cluster',
        xaxis_title='Clusters',
        yaxis_title='Area',
    )
    if plot_plotly_web:
        py.plot(fig, filename=f"surface_clusters", auto_open = True)
        fig.write_image(f"surface_clusters.png")

    st.plotly_chart(fig)

    
def plot_resultados_clustering(nclusters: int = 6, 
                               centro: str = 'CSR',
                               plot_plotly_web: bool = False) -> Dict[str, str]:

    
    n_clusters = int(nclusters)

    df_cluster = pd.read_parquet(f'data_for_app/data_cluster_trend_{centro}.parquet')
    df_toplot = df_cluster[['lat', 'lon', f'clusters_{n_clusters}']].copy()
    df_toplot[f'clusters_{n_clusters}'] = df_toplot[f'clusters_{n_clusters}'].astype('category')
    df_toplot[f'clusters_{n_clusters}'] = df_toplot[f'clusters_{n_clusters}'].apply(lambda x: 'cluster_'+str(x))
    
    df_toplot.to_csv(f'data_for_app/df_toplot_colab_clusters_{n_clusters}.csv', index=False)
    diccionario_colores = {k: v for k, v in zip(df_toplot[f'clusters_{n_clusters}'].unique(), px.colors.qualitative.Plotly)}
    if plot_plotly_web:
        df_toplot = df_toplot.query(f'clusters_{n_clusters}!="cluster_1"')
        df_toplot = df_toplot.query(f'clusters_{n_clusters}!="cluster_4"')
    st.markdown('Visualizamos los clusters en el mapa')
    fig = go.Figure()
    for cluster in df_toplot[f'clusters_{n_clusters}'].unique():
        mini_df = df_toplot.query(f'`clusters_{n_clusters}`==@cluster')

        fig.add_trace(go.Scattergeo(
            lon=mini_df["lon"],
            lat=mini_df["lat"],
            mode="markers",
            marker=dict(size=5, color=mini_df[f'clusters_{n_clusters}'].map(diccionario_colores)),
            text=mini_df[f'clusters_{n_clusters}'],
            name=mini_df[f'clusters_{n_clusters}'].unique()[0],
            showlegend=True,
            opacity=0.6
        ))

        # Customize the layout
        fig.update_geos(projection_type="orthographic",
                        showland=True,
                        landcolor="white",
                        showocean=True,
                        oceancolor="azure",
                        #oceancolor="aliceblue",
                        showrivers=True,
                        rivercolor="lightblue",
                        showlakes=True,
                        lakecolor="lightblue",
                        showcoastlines=True,
                        coastlinecolor="black",
        )

    fig.update_layout(
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
            ),
        geo=dict(
            scope="world",
            showland=True,
        ),
        width= 800, 
        height=800, 
        margin={"r":0,"t":0,"l":0,"b":0}
    )
    if plot_plotly_web:
        py.plot(fig, filename="plotly_geoscatter", auto_open = True)

    st.plotly_chart(fig)
    return diccionario_colores


def plot_grace_picture():    
    image = Image.open('grace.jpeg')
    st.sidebar.title('Satelite Grace')
    st.sidebar.image(image)



COLOR_MAP = {"default": "#262730",
             "pink": "#E22A5B",
             "purple": "#985FFF",}

def texto(texto : str = 'holi',
          nfont : int = 16,
          color : str = 'black',
          line_height : float =None,
          sidebar: bool = False):
    
    if sidebar:
        st.sidebar.markdown(
                body=generate_html(
                    text=texto,
                    color=color,
                    font_size=f"{nfont}px",
                    line_height=line_height
                ),
                unsafe_allow_html=True,
                )
    else:
        st.markdown(
        body=generate_html(
            text=texto,
            color=color,
            font_size=f"{nfont}px",
            line_height=line_height
        ),
        unsafe_allow_html=True,
        )
    

def generate_html(
    text,
    color=COLOR_MAP["default"],
    bold=False,
    font_family=None,
    font_size=None,
    line_height=None,
    tag="div",
):
    if bold:
        text = f"<strong>{text}</strong>"
    css_style = f"color:{color};"
    if font_family:
        css_style += f"font-family:{font_family};"
    if font_size:
        css_style += f"font-size:{font_size};"
    if line_height:
        css_style += f"line-height:{line_height};"

    return f"<{tag} style={css_style}>{text}</{tag}>"


    
if __name__ == "__main__":
    main()