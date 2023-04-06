#####################################################################

def json_columns(data, column):
    import pandas as pd
    data1 = data.copy()
    import json as j
    json = ''
    for x in data[column]:
        string = x      
        string = string.replace('%;  ', ',"',)
        string = string.replace('%; ', ',"',)
        string = string.replace('%;', '},',)
        string = string.replace(' ', '":',)
        string = '{\"'+ string
        json += string
    
    json = '['+ json + ']'
    json = json.replace('},]', '}]')    
    json = pd.DataFrame.from_dict(j.loads(json), orient='columns')
    try:
      data1.drop(columns = [column, '!CPS Failed'], inplace=True)
    except:
      pass
    return pd.concat([data1, json], axis = 1)

#####################################################################


def remove_chars(data):
  main_df = data.copy()
  for i in range(0,5):
    for i, v in enumerate(main_df.columns):
      string = str(v)
      main_df.rename(columns = {v:string.lower()}, inplace = True)

    for i, v in enumerate(main_df.columns): #por algum motivo se vc fizer rename no mesmo 'for' nao funciona
      string = str(v)
      main_df.rename(columns = {v:string.strip()}, inplace = True)

    for i, v in enumerate(main_df.columns):
      string = str(v)
      main_df.rename(columns = {v:string.strip('%')}, inplace = True)

    for i, v in enumerate(main_df.columns):
      string = str(v)
      main_df.rename(columns = {v:string.strip('_')}, inplace = True)
  print(f'Check headers conversion: \n \n {main_df.columns}')
  return main_df

#####################################################################

def upload(files):
  tables = dict()
  for i, n in enumerate(files):
      print(n)

