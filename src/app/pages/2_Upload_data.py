import streamlit as st
from PIL import Image
import pandas as pd
import time
import sys
sys.path.append('./src')

# append folder to path in order to import functions
import functions_ as f


icon = Image.open('./docs/if-crystal-shard-2913097_88819.png')
st.set_page_config(
    page_title = 'Mineral Clustering',
    page_icon= icon)

# Instructions
st.markdown('''
#### It is recommended that the files have an informative and brief name in order to concatenate all csv files. To recognize the samples, that name will be added as a new feature column.

### **You can select multiples files to upload:**
''')
st.markdown('e.g. [ XY_150.csv, XY_90.csv, XY_-20.csv ], where XY refers to the sample itself, and the code refers to the granulometric classification.')


# Upload
uploaded_files = st.file_uploader('',accept_multiple_files=True)

# import tables with pandas

tables = dict()
for i in uploaded_files:
    sample = (i.name[:-4])
    tables[sample] = pd.read_csv(i)

# processing EDS column

file_list = []
for i in tables.keys():
    file_list.append(i)
    tables[i] = f.json_columns(tables[i], 'EDS Normalised')
    tables[i].drop('EDS Normalised', axis = 1, inplace = True)
    tables[i].insert(loc = 0, column = 'sample', value = i)
    # st.table(tables[i].head(3))


if len(file_list) > 0:

# concatenating all files in one dataframe

    concat_data = pd.concat([tables[key] for key in file_list], axis = 0, ignore_index=True)

    st.markdown('This is the total amount of rows for each file:')
    st.table(
    concat_data.groupby('sample').size().reset_index(name='count')
    )

    if st.button('Check sample from table'):
        st.table(concat_data.sample(20))

# Create a download button

    download_name = f'{time.localtime()[0]}-{time.localtime()[1]}-{time.localtime()[2]}-{time.localtime()[4]}'

    st.download_button('Download', 
                    data = concat_data.to_csv(sep = ';', index=False).encode('utf-8'),
                    file_name = f'data_{download_name}.csv')

    st.markdown('**Now we are done here, you already have a dataset to proceed to Clustering page.**')

# SESSION STATE
    # st.session_state
    if 'df' not in st.session_state:
        st.session_state['df'] = concat_data

    # st.table(st.session_state.df.head(10))