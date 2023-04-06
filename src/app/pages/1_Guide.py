import streamlit as st
from PIL import Image

icon = Image.open('./docs/if-crystal-shard-2913097_88819.png')
st.set_page_config(
    page_title = 'Mineral Clustering',
    page_icon= icon)

text = '''
Mineralogic Explorer
Exporting .csv with EDS data.
'''

st.markdown(text)