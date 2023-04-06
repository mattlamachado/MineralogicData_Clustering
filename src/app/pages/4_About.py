import streamlit as st
from PIL import Image
import requests

icon = Image.open('./docs/if-crystal-shard-2913097_88819.png')
st.set_page_config(
    page_title = 'Mineral Clustering',
    page_icon= icon)

# ABOUT
st.markdown('''## About
Hello, my name is Matheus Machado, I designed this aplication to simplify Automated Mineralogy analyses.
This is the result of our efforts to help technicians dealing with a variety of materials, 
facing the challenge of defining mineral chemistry thresholds and identifying phases in smaller proportions on 
[Mineralogic](https://www.zeiss.com/microscopy/en/products/software/zeiss-mineralogic.html).

I want to thanks the valuable feedbacks from our co-workers regarding interface and user experience and infrastructure support from [CETEM](https://lmct.online). 
Our goal is to make it accessible and user-friendly for everyone.
If you have any questions, issues or tips for us, please don't hesitate to contact me.

mattlamachado@gmail.com

''')

# SOCIAL MEDIA

col1, col2, col3 = st.columns([0.15,0.15,2])

with col1:
    st.markdown('<a href="https://www.linkedin.com/in/mattlamachado/"><img src="https://cdn3.iconfinder.com/data/icons/social-media-2285/512/1_Linkedin_unofficial_colored_svg-512.png" width="30"></a>', unsafe_allow_html=True)

with col2:
    st.markdown('<a href="https://github.com/mattlamachado"><img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="30"></a>', unsafe_allow_html=True)

with col3:
    response = requests.get("https://api.github.com/users/mattlamachado")

    if response.status_code == 200:
        user_data = response.json()
        
    # Display GitHub profile photo with circular border
        st.markdown(f'''<div style="float:right"><img src="{user_data["avatar_url"]}" alt="GitHub profile photo" style="border-radius: 50%; width: 150px;"></div>
        ''', unsafe_allow_html=True)

    else:
        st.write("Error retrieving data from GitHub API")