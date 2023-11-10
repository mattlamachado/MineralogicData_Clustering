import streamlit as st
from PIL import Image

icon = Image.open('./docs/if-crystal-shard-2913097_88819.png')
st.set_page_config(
    page_title = 'Mineral Clustering',
    page_icon= icon)

text1 = '''
### **Upload Data**  
You can import multiples csv files, it’s suggested that the file name contains information about the sample, because it will be incorporated 
as a column on the concatenated table.
The app first was designed to receive EDS data from the output of Mineralogic, so with Mineralogic Explorer you can export grain data on Particle data sheet (**Fig. 1**); 

But now you can upload any table like csv file separated by simple comma, and cluster whatever you want.


On **Fig. 1.1**, you can select what features to export, the app is designed to extract chemistry data from  “EDS Normalised” columns, 
which means every element information is stored in this column. Columns with special characters may crash the app (“!CPS Failed”), 
unselect this feature, or delete from csv file. On **Fig 1.2** you can export the csv file, which may feed the MineralClustering application.'''

text2='''
The user may want to select features that can help differentiate phases, although it was designed for chemistry data, 
you can try to input morphological data to the clustering, such as roughness, area, or porosity for example.
If you’re pretty knowing what you’re looking for, you can restrict the data to analyse specific compositions or phases. 
Mineralogic Explorer provides filters that can be used before the exportation, where you can select previously defined phases (**Fig. 3**), 
or restrict chemistry, in order to assess clusterization of more specific composition ranges (**Fig. 2,3**).'''

text3='''
On Upload Data page, you can select multiple csv files, and check if the importation was successful, and eventually download the concatenated dataset. 
Now you may proceed to Clustering page. As the app uses a free Streamlit application, it limits some processing, 
that’s why you should resample randomly a fewer number of rows to cluster, both avoiding long processing time and avoiding crashing the app. 
Here you need to select which features will be considered by the model to cluster, the model requires no constant variables, so make sure to select elements, 
or features that have values different from NaN values or zeros.

After running the iteration of different number of clusters, you’ll face some metrics regarding the quality of the clusterizations, 
and can make a decision about how many clusters are more adequate to your data. You can select the proper number and proceed to data analysis.

### **PCA visualization**  
Principal Component Analysis (PCA) is a statistical method used in data analysis to reduce the dimensionality of a dataset while retaining as much of the 
variance in the data as possible. It works by transforming the original variables into a new set of orthogonal variables called principal components. 
These principal components are linear combinations of the original variables, and they are ordered in such a way that the first component explains 
the most variance, the second explains the second most, and so on. By retaining only a subset of these principal components, you can effectively reduce 
the dimensionality of the data while preserving its essential information.

The advantages of using PCA are twofold. Firstly, it simplifies complex datasets by reducing the number of variables, making it easier to visualize 
and interpret the data. This can be particularly useful in various fields such as image processing, mineral chemistry, and finance, where dealing with 
high-dimensional data is common. Secondly, PCA helps in identifying underlying patterns and relationships in the data by emphasizing the most significant 
sources of variation. This can lead to improved data understanding, better feature selection, and more efficient modeling in machine learning or other 
data analysis tasks.
This is a fast way to check if your clusterization has any evident characteristics of a really bad result, 
it isn’t able to provide all information about all clusters, but you can check if the PCs are somehow clustered properly. 
You’ll be able to observe some points concentrations already classified, as a unique cluster.

### **Plots**  
A scatterplot and histogram are displayed, where you can select the elements to plot, and clusters. 
It may help you to assess the different characteristics of each cluster, and understand why such cluster was classified differently from another (**Fig. 4**).

At the end, a summary table, it provides some statistics like mean, median, min and max, and percentiles for each feature modeled and for each cluster. 
This information is all about defining the thresholds for each phase of automated classification on Zeiss Mineralogic.
'''

fig1_legend = '<center>Figure 1. Particle Data sheet on Mineralogic Explorer.</center>'
fig2_legend = '<center>Figure 2. Filtering grain data by phases.</center>'
fig3_legend = '<center>Figure 3. Filtering grain data by range of feature, in this case the range of Ti grade on minerals.</center>'
fig4_legend = '<center>Figure 4. Clustering of three phases and its respective chemical distribution.</center>'

fig2_img_filter_mineral = Image.open('./docs/filter_bymineral_.png')
fig3_img_filter_chem = Image.open('./docs/filter_bychem_.png')
fig1_img_table = Image.open('./docs/table_.png')
fig4_clustering = Image.open('./docs/clustering.png')

st.markdown(text1)
st.image(fig1_img_table)
st.markdown(fig1_legend, unsafe_allow_html=True)
st.markdown('  ')
st.markdown(text2)
st.image(fig2_img_filter_mineral)
st.markdown(fig2_legend, unsafe_allow_html=True)
st.markdown('  ')
st.image(fig3_img_filter_chem)
st.markdown(fig3_legend, unsafe_allow_html=True)
st.markdown('  ')
st.markdown(text3)
st.image(fig4_clustering)
st.markdown(fig4_legend, unsafe_allow_html=True)


