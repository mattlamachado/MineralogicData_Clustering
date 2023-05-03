import streamlit as st
from PIL import Image
from sklearn.preprocessing import StandardScaler, FunctionTransformer, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from scipy.stats import gmean
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
sys.path.append('./src')

# append folder to path in order to import functions
from functions_ import *

icon = Image.open('./docs/if-crystal-shard-2913097_88819.png')
st.set_page_config(
    page_title = 'Mineral Clustering',
    page_icon= icon,
    layout="wide")

# PLOT SETTINGS
font = {'family' : 'arial',
        'size'   : 12}
plt.rc('font', **font)

if not st.session_state:
    st.markdown('### Go to Upload data page first!')

# START SESSION _____________________________________________________________________
if st.session_state:

    X = st.session_state.df

    col1, col2 = st.columns(2)

    with col1:
        elements = st.multiselect('Pick elements to cluster:', X.columns)

    with col2:
        # sampling to reduce modeling time
        n_samples = st.number_input('number of samples', min_value=100, value=1000)
    
    if elements:
        X.fillna(10**-5, inplace=True) # 10**-5
        X.replace(0,10**-5, inplace=True,)
        x = X.loc[:,elements].sample(n=n_samples, random_state=42, ignore_index=True)

# tuning of components
    CLR = FunctionTransformer(clr)

    st.markdown('## Discover the best number of components:')
    st.markdown('''This may take a while, so stay aware of the parameters you choose. 
    The sample's size and number of components can impact sigficatively the processing time.''')

    run_model = st.button('Run clusterization') 
        
    if run_model:
        metrics = pd.DataFrame()

        # tuning of n-components _____________________________________________________
        index = 0
        n_min, n_max, step = 10, 30, 1

        # PROGRESS
        progress_bar = st.progress(0)
        progress_step = len(np.arange(n_min, n_max, step))
        # ____________________________________________________________________________

        for n in np.arange(n_min, n_max, step):
            for tol in [1]:
                pipeline = Pipeline(steps = 
                [
                    ('powerTransformer', PowerTransformer(method='box-cox', standardize=True)),
                    # ('centered log-ratio', CLR), # centered log-ratio transformation
                    # ('standard scaler', StandardScaler()),
                    ('model', GaussianMixture(
                                        n_components=n,
                                        tol=tol, max_iter = 2000, random_state=42
                                        ))
                ])
                
                model_ = pipeline.fit(x)
                classification = model_.predict(x)

                
                metrics.loc[index, ['n', 'tol', 'calinski', 'bouldin', 'silhouette', 'BIC', 'AIC']] = [
                                                                                    n, tol, 
                                                                                    calinski_harabasz_score(x, classification), 
                                                                                    davies_bouldin_score(x, classification), 
                                                                                    silhouette_score(x, classification),
                                                                                    model_['model'].bic(x), model_['model'].aic(x)
                                                                                    ] 
                index+=1
                progress_bar.progress(index/progress_step)
        #______________________________________________________________________________

        # ploting metrics

        a=1
        metrics_fig = plt.figure(figsize=(15, 7), dpi = 300)
        for metric in ['calinski', 'bouldin', 'silhouette', 'BIC', 'AIC']:
            plt.subplot(2,3,a)
            a+=1
            sns.lineplot(data=metrics,
                        y=metric,
                        x='n')
            plt.grid(which='both', color='grey', linewidth=0.5)
            plt.xticks(np.arange(n_min, n_max, step)) #, minor=True
        plt.suptitle('Metrics')
        metrics_fig.tight_layout()
        if metrics_fig not in st.session_state:
            st.session_state['metrics_fig'] = metrics_fig
    
    if 'metrics_fig' in st.session_state:
        st.pyplot(st.session_state['metrics_fig'])

    st.markdown('''**Silhouette Score**: This metric measures how well each data point in a cluster is separated from other clusters. 
    The score ranges from -1 to 1, with higher values indicating better cluster separation. A score close to 1 indicates well-defined clusters, 
    while a score close to 0 indicates overlapping clusters, and negative scores indicate that data points may have been assigned to the wrong cluster.
    As GaussianMixture considers overlaping of clusters, silhouette presents low values.
    \n**Davies-Bouldin Score**: This metric measures the average similarity between each cluster and its most similar cluster. A lower score indicates better 
    clustering, with 0 indicating perfectly separated clusters. Higher scores indicate that clusters are overlapping or not well-separated.
    \n**Calinski-Harabasz Score**: This metric measures the ratio of between-cluster variance to within-cluster variance. Higher scores indicate 
    better clustering, with a score close to 1 indicating well-separated clusters. Lower scores indicate overlapping or poorly separated clusters.
    \n**BIC and AIC** are both model selection criteria that balance a model's goodness of fit to the data with its complexity. 
    A lower score indicates a better model fit. The BIC places a higher penalty on model complexity than the AIC, so it tends to favor simpler models. 
    In general, the model with the lowest BIC score is preferred. The AIC, on the other hand, may favor more complex models that fit the data well.
    When using these criteria, it's important to keep in mind that they are relative measures of model quality, 
    so the scores themselves do not have any absolute meaning. Instead, the scores are used to compare different models and choose the best one 
    for the problem at hand.''')

    # MODELING ________________________________________________________________________

    st.markdown('## Modeling')
    clusters = st.slider('Number of components:', min_value=2, max_value=80, step=1 )

    pipeline_transformer = Pipeline(steps = 
    [
        # ('centered log-ratio', CLR), # centered log-ratio transformation
        # ('standard scaler', StandardScaler())
        ('powerTransformer', PowerTransformer(method='box-cox', standardize=True))
    ])

    pipeline = Pipeline(steps= 
    [
        ('transformer', pipeline_transformer),
        ('model', GaussianMixture(
            n_components=clusters,
            tol=1,max_iter = 600, random_state=42))
    ])

    # st.text(pipeline)

    # PLOT RESULTS AS PCS _______________________________________________________________

    st.markdown('''The following visualization is a manner of assaying the results of clustering through dimensionality-reduction. 
    The actual clustering considered the n-elements dimensions. The output results can be altered by many factors, such as: 
    type of distribution for each element, nucleation of phases, superposition among phases, scarcity of a specific phase. 
    It's recommended to go further on [GaussianMixture](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html) 
    to understand better the results.
    ''')

    classification = pipeline.fit_predict(x)

    pcs = pd.DataFrame(data=PCA(4).fit_transform(x), columns=[1,2,3,4])
    graphs = [[1,2], [1,3], [2,3], [1,4]]

    pca_fig = plt.figure(dpi=300)
    a = 1
    for g in graphs:
        plt.subplot(2,2,a)
        a+=1
        sns.scatterplot(x = pcs.loc[:, g[0]],
                        y = pcs.loc[:, g[1]],
                        s = 2,
                        hue = classification,
                        palette = 'tab10',
                        alpha = 0.5)
        plt.legend([],[], frameon=False)
        # plt.yscale('log')
        # plt.xscale('log')  
    plt.suptitle('Principal Component visualization')
    plt.tight_layout()

    st.pyplot(pca_fig)

    colx, coly, colclass = st.columns([1,1,3])

    with colx:
        x_ = st.selectbox('x axis',elements)
    with coly:
        y_ = st.selectbox('y axis', elements)
    with colclass:
        selected_classes = st.multiselect('Classes', np.unique(classification))

    if not selected_classes:
        selected_classes = np.unique(classification)
    
    class_filter = [c in selected_classes for c in classification] 

    scatter_fig = plt.figure(dpi=200)
    sns.scatterplot(
        data = x[class_filter],
        x = x_, y = y_, hue = classification[class_filter], 
        palette='tab10', s=4, alpha=0.7
    )
    
    st.pyplot(scatter_fig)



    # full dataframe
    # classification_full = pipeline['centered log-ratio'].transform(X.loc[:, elements].fillna(10**-5))
    # classification_full = pd.DataFrame(data = pipeline['standard scaler'].transform(classification_full), columns = X[elements].columns)
    # classification_full['class'] = pd.Series(pipeline.predict(X.loc[:, elements].fillna(10**-5)))

    # aggregation = ['min', 
    #               ('M-3*SD', lambda x: np.mean(x) - 3*np.std(x, ddof=1)),
    #               ('2%', lambda x: np.percentile(x, q=0.02)),
    #               ('10%', lambda x: np.percentile(x, q=0.1)), 
    #               'median', 
    #               'mean',
    #               ('90%', lambda x: np.percentile(x, q=0.9)), 
    #               ('98%', lambda x: np.percentile(x, q=0.98)),
    #               ('M+3*SD', lambda x: np.mean(x) + 3*np.std(x, ddof=1)), 
    #               'max']
    
    aggregation_dict = {
        'min':'min',
        'M-3*SD': lambda x: (np.mean(x) - (2*np.std(x, ddof=1))),
        '1%': lambda x: np.percentile(x, q=1),
        '2%': lambda x: np.percentile(x, q=2),
        '10%': lambda x: np.percentile(x, q=10),
        # '50%': lambda x: np.percentile(x, q=50),
        'median': lambda x: np.median(x),
        'mean': lambda x: np.average(x),
        # 'stdev': lambda x: np.std(x, ddof=1),
        '90%': lambda x: np.percentile(x, q=90),
        '98%': lambda x: np.percentile(x, q=98),
        '99%': lambda x: np.percentile(x, q=99),
        'M+3*SD': lambda x: (np.mean(x) + (2*np.std(x, ddof=1))),
        'max':'max'
    }

    classification_full = x.copy() 
    # pd.DataFrame(data= pipeline_transformer.transform(x), columns = x.columns)
    classification_full['class'] = classification

    summary_dict = dict()
    for i in aggregation_dict.keys():
        summary_dict[i] = classification_full.groupby(by='class')[elements].agg(aggregation_dict[i])
        # st.table(summary_dict[i])
        # summary_dict[i] = pd.DataFrame(data = pipeline_transformer.inverse_transform(summary_dict[i]), columns=summary_dict[i].columns)
        # st.table(summary_dict[i])

    final_df = pd.DataFrame()
    for col in x.columns:
        for key in summary_dict.keys():
            col_name = f'{col}_{key}'
            final_df[col_name] = summary_dict[key][col]

    st.table(final_df)

    import time
    download_name = f'{time.localtime()[0]}-{time.localtime()[1]}-{time.localtime()[2]}-{time.localtime()[4]}'

    st.download_button('Download', 
                    data = final_df.to_csv(sep = ';', index=True).encode('utf-8'),
                    file_name = f'summary_table_{clusters}clus_{n_samples}n_{download_name}.csv')
    

