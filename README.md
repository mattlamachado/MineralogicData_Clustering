# MineralogicData_Clustering
Mineral chemistry based clustering.

```bash
conda create -n stenv python=3.9.12
conda activate stenv
pip install -r requirements.txt
streamlit run Home.py
```

Based on a preassumption that after the preprocessing of data, throught a pipeline that includes a powertransformation and a standardization, it's expected that for different datasets the same binwidth should work well. 

