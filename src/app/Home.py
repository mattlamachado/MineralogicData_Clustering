import streamlit as st
from PIL import Image

icon = Image.open('./docs/if-crystal-shard-2913097_88819.png')
st.set_page_config(
    page_title = 'Mineral Clustering',
    page_icon= icon)

bg = Image.open('./docs/automated_classification_minerals.png')
st.image(bg)

st.markdown('''# **Mineral Clustering for Zeiss Mineralogic**
This aplication was developed to assist 
[Zeiss Mineralogic](https://www.zeiss.com/microscopy/en/products/software/zeiss-mineralogic.html) 
analysis mineral classification by combining EDS data from spot centroid or mapping in .csv 
into a pipeline of compositional data preprocessing and unsupervised clustering.
---''')

