def import_all():
    print('''import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import numpy as np
    import pandas as pd''')

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

def ternary_plot(x, y, z, color = None, 
                 non_num = None, discrete_colors = {}, 
                 barvar = None, cmapcolor = None, vmin = None, vmax = None,
                 label = False, grid = False, 
                 fields = None, frame = (10,10),
                 axis_label=None, s=120, title=None,
                 title_fontsize=12, fontsize=12, frameon=False,
                 ):
  ''' 
  ternary plot, first of all, normalizes the 3D coordinates, and 
  re-normalizes x,y coordinates between themselves
  so, it passes the normalized data as (X,Y) coordinates to be plotted 
  within a triangle
  
  color:  a unique color for every data

  non_num:  a non-numeric variable that can be classified by a discrete list 
            of colors

  discrete_colors:  a dictionary that carries each unique value of 
                    non_num *arg with its respective color
  
  barvar: the column(var) that goes on the continuous colorbar
            e.g., a fourth element

  vmax, vmin: max and min of color bar

  cmapcolor:  cmap pattern ('viridis', 'plasma', 'inferno', 'magma', 'cividis', 
              'jet', 'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone',
              'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
              'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper')

  label:  which column data to be plotted as a label por point

  grid: default = False

  fields: 'chromite' for chromite classification ternary (Cr, Al, Fe)
          'asbolane' for asbolane-lithiphorite ternary (Al, Ni, Co)
  
  frame: sets the size of the output image
  
  axis_label: receives a tuple with the Axes' names to be writen on the ternary, 
  where: (x,y,z) --> (left, right, upper); the default will return the Column.name

  s: float or array-like, shape (n, ), optional
  The marker size in points**2

  LEGEND:
  title: string
  title_fontsize: title
  frameon: boolena to plot a frame outside legend
  fontsize: size of the legend description itself.
  shadow: shadow under the legend
  '''
  ### DEFINING NORMALIZATION FUNC ###
  def normalization(x,y,z):
      sum_norm = []
      x_norm = []
      y_norm = []
      z_norm = []

      for i in range(0, len(x)):
        sum_norm.append(x[i] + y[i] + z[i])

      for i in range(0, len(x)):
        x_norm.append(x[i]/sum_norm[i])
        y_norm.append(y[i]/sum_norm[i])
        z_norm.append(z[i]/sum_norm[i])
      return x_norm, y_norm, z_norm

      #print(x[0], y[0],  z[0], sum_norm[0])

  import matplotlib.pyplot as plt
  import matplotlib.lines as lines
  import matplotlib.patches as mpatches
  
  ### NORMALIZATION OF 3D COORDINATES ###
  x_norm, y_norm, z_norm = normalization(x,y,z)
  
  ### 3D to 2D coordinates ###
  X = []
  Y = []
  X.clear()
  Y.clear()
  for i in range(0,len(z_norm)):
    Y.append(z_norm[i])

  for i in range(0, len(x_norm)):
    X.append(((y_norm[i]/(x_norm[i] + y_norm[i])) * (1-Y[i])) + 0.5*Y[i])
      # normalizing x n y again for the X axis, multiply for the complement of 
      # Y axis to adjust to triangle, so add a displacement proportional to Y

  ### setting some ternary ploting configs like ternaty height h 
  h = (((3**(1/2)))/(2))+0.012

  #print(h)

  ### PLOT SIZE ###
  if cmapcolor != None:
    fig = plt.figure(figsize =(frame[0]+2,frame[1]*h))
    fig, ax = plt.subplots(1, figsize =(frame[0]+2,frame[1]*h), facecolor='white')
  else:
    fig = plt.figure(figsize =(frame[0],frame[1]*h))
    fig, ax = plt.subplots(1, figsize =(frame[0],frame[1]*h), facecolor='white')
  
  ### TERNARY BOARDER ###
  triangle = ((0.0, 0.5, 1, 0), (0.001, 1, 0.001, 0.001))
  ax.plot(triangle[0],triangle[1], '-', color = 'black', linewidth = '1.2')

  ### TERNARY GRID ###
  if grid == True:
    grid = {'hgrid' : ((0.05, 0.95, 0.9, 0.1, 0.15, 0.85, 0.8, 0.2, 0.25, 0.75, 0.7, 0.3, 0.35, 0.65, 0.6, 0.4, 0.45, 0.55),
                    (0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5, 0.6, 0.6, 0.7, 0.7, 0.8, 0.8, 0.9, 0.9)),
            'xgrid' : ((0.05, 0.1, 0.2, 0.1, 0.15, 0.3, 0.4, 0.2, 0.25, 0.5, 0.6, 0.3, 0.35, 0.7, 0.8, 0.4, 0.45, 0.9, 1),
                    (0.1, 0, 0, 0.2, 0.3, 0, 0, 0.4, 0.5, 0, 0, 0.6, 0.7, 0, 0, 0.8, 0.9, 0, 0)),
            'ygrid' : ((0.1, 0.55, 0.6, 0.2, 0.3, 0.65, 0.7, 0.4, 0.5, 0.75, 0.8, 0.6, 0.7, 0.85, 0.9, 0.8, 0.9, 0.95),
                    (0, 0.9, 0.8, 0, 0, 0.7, 0.6, 0, 0, 0.5, 0.4, 0, 0, 0.3, 0.2, 0, 0, 0.1))}
    for i in grid.keys():
      ax.plot(grid[i][0], grid[i][1], '-',color = 'black', linewidth = '0.2', alpha = 0.5)

  ### CHROMITE FIELDS ###

  if fields == 'chromite':
    hfields = ((0.125, 0.875, 0.75, 0.25, 0.375, 0.625),
              (0.25, 0.25, 0.5, 0.5, 0.75, 0.75))
    ax.plot(hfields[0], hfields[1], '--',color = 'black', alpha = 0.7, linewidth = '1')

    vfields = ((0.25, 0.375, 0.625, 0.75, 0.5, 0.5),
              (0, 0.25, 0.25, 0, 0, 0.75))
    ax.plot(vfields[0], vfields[1], '--',color = 'black', alpha = 0.7, linewidth = '1')
    
    field_label = (('Magnetite', 'Cr-magnetite', 'Al-magnetite', 'Ferrian chromite', 'Ferrian pickotite', 'Pickotite', 'Hercynite', 'Chromite', 'Al-chromite'),
                   (0.5, 0.4, 0.6, 0.37, 0.63, 0.6, 0.82, 0.18, 0.4),
                   (0.85, 0.6, 0.6, 0.4, 0.4, 0.1, 0.1, 0.1, 0.1,))

    for i, l in enumerate(field_label[0]):
        ax.annotate(l, xy=(field_label[1][i], field_label[2][i]), xytext=(field_label[1][i], field_label[2][i]),
                  arrowprops=None, ha='center', va='center', rotation= 0, size=12)
        
  ### ASBOLANE-LITHIOPHORITE FIELDS ###

  if fields == 'asbolane':
    asbolane = ((0.45, 0.9, 1, 0.75, 0.675), (0.9, 0, 0,0.5, 0.45))
    ax.plot(asbolane[0], asbolane[1], '-',color = 'black', linewidth = '1')

    ax.annotate('Lithiophorite', xy=(0.08, 0.05), xytext=(0.3, 0.2),
                arrowprops=dict(arrowstyle='->'), ha='center', va='center', rotation= 0, size=15)
    ax.annotate('Co-rich', xy=(0.6, 0.7), xytext=(0.6, 0.7),
                arrowprops=None, ha='center', va='center', rotation= -60, size=15)
    ax.annotate('Ni-rich', xy=(0.83, 0.24), xytext=(0.83, 0.24),
                arrowprops=None, ha='center', va='center', rotation= -60, size=15)
    ax.annotate('Asbolane', xy=(0.925, 0.24), xytext=(0.78, 0.53),
                arrowprops=None, ha='center', va='center', rotation= -60, size=15)
    
  ### MG SILICATES FIELDS ###

  if fields == 'mgsilicates':
    mgsilicates = ((0.875, 0.215, 0.305, 0.81, 0.8, 0.315), (0.25, 0.43, 0.61, 0.38, 0.4, 0.635))
    ax.plot(mgsilicates[0], mgsilicates[1], '--',color = 'black', linewidth = '1')

    ax.annotate('serpentine series', xy=(0.08, 0.05), xytext=(0.23, 0.425),
                arrowprops=None, ha='left', va='center', rotation= -13, size=8)
    ax.annotate('talc series', xy=(0.08, 0.05), xytext=(0.31, 0.575),
                arrowprops=None, ha='left', va='center', rotation= -21, size=8)
    ax.annotate('sepiolite series', xy=(0.08, 0.05), xytext=(0.33, 0.62),
                arrowprops=None, ha='left', va='center', rotation= -22, size=8)


  # points label #
  if bool(label) == True:
    annotations = label
    for i, label in enumerate(annotations):
        plt.annotate(label, (0.02 + X[i], 0.02 + Y[i]), fontsize = 14, weight = 'normal')

  ### CONFIG ###
  plt.ylim(0, 1)
  plt.xlim(0, 1)
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  plt.axis('off')  # command for hiding the axis.


  ### ANNOTATE ###
  # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.annotate.html
  # annotate axis labels #
  if axis_label != None:
    axis_lab = ((axis_label[0], axis_label[1], axis_label[2]),
                  (0, 1, 0.5),
                  (-0.04, -0.04, 1.035))
  else:
    axis_lab = ((x.name, y.name, z.name),
                  (0, 1, 0.5),
                  (-0.04, -0.04, 1.035))


  for i, l in enumerate(axis_lab[0]):
    ax.annotate(l, xy=(axis_lab[1][i], axis_lab[2][i]), xytext=(axis_lab[1][i], axis_lab[2][i]),
              arrowprops=None, ha='center', va='center', rotation= 0, size=20, annotation_clip = False)

  ### PLOT ###
  if color != None:
    plt.scatter(X, Y, s = s, alpha = 1, c = color, edgecolors = 'black')

  elif type(non_num) != type(None):
    plt.scatter(X, Y, s = s, alpha = 1, c = non_num.map(discrete_colors), edgecolors = 'black')
    patches = [ mpatches.Patch(color=c, label=g) for g,c in discrete_colors.items() ]
    plt.legend(handles=patches, loc='upper left', ncol=1, frameon = True, 
               title = title, title_fontsize = title_fontsize, edgecolor = 'black', fontsize = fontsize, shadow=True) #bbox_to_anchor=(0.1,0.9)


  elif cmapcolor != None:
    plt.scatter(X, Y, s = s, alpha = 1, c = list(barvar), cmap = cmapcolor , edgecolors = 'black', vmax=vmax, vmin=vmin) # 'inferno' 'jet'
    cb = plt.colorbar()
    cb.set_label(barvar.name, color= 'black', fontsize = 18)

  else: 
    plt.scatter(X, Y, s = s, alpha = 1, c = 'black', edgecolors = 'black')
    


  ### SAVE FIG ###
  #save_fig = input('save? (y/n)')
  #if save_fig == 'y':
  #  fname = input('file name: ')
  #  plt.savefig(f'{fname}.png', transparent=False)
  #  files.download(f'{fname}.png')
  #else: pass

  plt.show()

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

def elem_to_oxide(main_df, conversion_df):
  """Converts dataframe's header and values using a conversion sheet as criteria."""
  import pandas as pd
  new_df = main_df.copy() # new dataframe to delete data before the remove_chars() to maintain uppercases etc

  ## clean headers ##
  main_df = remove_chars(main_df) # check headers ...

  ##################################
  
  ## creating new empty dataframe ##
  breakFlag = False
  for i, v in enumerate(main_df.columns): 
    for j in conversion_df['Elem']:
      # print(f'{v} == {j.lower()} = {v == j.lower()}')
      if v == j.lower():                                # testing if the column name matches any elements
        to_delete = new_df.iloc[:, i:].columns.tolist() # setting list of columns to delete, based on first 
        new_df.drop(columns= to_delete, inplace=True)   # element column index [ i:]; dropping columns.
        breakFlag = True # --> all this bullshit is to speed the code, skipping both loops
        break            # otherwise the loop would keep testing if the column in main_df exists
    if breakFlag == True:
      break
    
  ## converting data to new dataframe ##      
  for i in main_df.columns:                             #same loop than before
    if main_df.loc[:, i].all() != 'NaN' or '' or 0. or None: # skip no data columns 
      for j, k, m in zip(conversion_df['Elem'], conversion_df['Oxide'], conversion_df['ElemToOx']): #DataFrame with headers now
        if i == j.lower():   # test if elem matches main_df columns, if True append new pd.Series to the new_df
                             # the thing if when you define a column that doesnt exist, it creates another one, it works like an append
          new_df[f'{k}'] = pd.Series(dtype = 'float16') # saying to new_df to set a new pd.Series named after the Oxide column (k), 
                                                        # so it creates a new empty one 
                                                        # dtype: defining the dtype to skip an output warning
          new_df.loc[:, k] = main_df.loc[:, i] * m      # finally taking the main_df value, multiplying by the 
                                                        # ElemToOx factor and putting to the values of the new column

  return new_df

#####################################################################

def oxide_to_elem(main_df, conversion_df):
  """Converts dataframe's header and values using a conversion sheet as criteria."""
  import pandas as pd
  new_df = main_df.copy() # new dataframe to delete data before the remove_chars() to maintain uppercases etc

  ## clean headers ##
  main_df = remove_chars(main_df) # check headers ...

  ##################################
  
  ## creating new empty dataframe ##
  breakFlag = False
  for i, v in enumerate(main_df.columns): 
    for j in conversion_df['Oxide']:
      # print(f'{v} == {j.lower()} = {v == j.lower()}')
      if v == j.lower():                                # testing if the column name matches any elements
        to_delete = new_df.iloc[:, i:].columns.tolist() # setting list of columns to delete, based on first 
        new_df.drop(columns= to_delete, inplace=True)   # element column index [ i:]; dropping columns.
        breakFlag = True # --> all this bullshit is to speed the code, skipping both loops
        break            # otherwise the loop would keep testing if the column in main_df exists
    if breakFlag == True:
      break
    
  ## converting data to new dataframe ##      
  for i in main_df.columns:                             #same loop than before
    if main_df.loc[:, i].all() != 'NaN' or '' or 0. or None: # skip no data columns 
      for j, k, m in zip(conversion_df['Oxide'], conversion_df['Elem'], conversion_df['OxToElem']): #DataFrame with headers now
        if i.lower() == j.lower():   # test if elem matches main_df columns, if True append new pd.Series to the new_df
                             # the thing if when you define a column that doesnt exist, it creates another one, it works like an append
          new_df[f'{k}'] = pd.Series(dtype = 'float16') # saying to new_df to set a new pd.Series named after the Oxide column (k), 
                                                        # so it creates a new empty one 
                                                        # dtype: defining the dtype to skip an output warning
          new_df.loc[:, k] = main_df.loc[:, i] * m      # finally taking the main_df value, multiplying by the 
                                                        # ElemToOx factor and putting to the values of the new column

  return new_df

#####################################################################

def oxide_to_mol(data, conversion_df, O, justAPFU=True):
    '''
    O : the basis amount of Oxigens of the mineral formula
    justAPFU : if True, it returns just a converted DataFrame
               if False, it returns cation_proportion, equivalent_charge, Oxigen%_factor, converted APFU 
    '''
    import pandas as pd
    new_df = data.copy()
    main_df = remove_chars(data)

    ## creating new empty dataframe ##
    breakFlag = False
    for i, v in enumerate(main_df.columns): 
        for j in conversion_df['Oxide']:
            if v == j.lower():
                index_first_elem = i
                to_delete = new_df.iloc[:, i:].columns.tolist()
                new_df.drop(columns= to_delete, inplace=True)
                breakFlag = True
                break
            if breakFlag == True:
                break
    
    eq_charge = new_df.copy()

    for i in main_df.columns:                             
        if main_df.loc[:, i].all() != 'NaN' or '' or 0.0 or None:
            for e, j, k, m, n in zip(conversion_df['Elem'], conversion_df['Oxide'], conversion_df['Cations'], 
                                        conversion_df['MolecMass'], conversion_df['Valency']):
                if i.lower() == j.lower():
                    new_df[f'{e}'] = pd.Series(dtype = 'float16')
                    new_df.loc[:, f'{e}'] = (main_df.loc[:, i] / m) * k # cation proportion

                    eq_charge[f'{e}'] = pd.Series(dtype = 'float16')
                    eq_charge.loc[:, f'{e}'] = new_df.loc[:, f'{e}'] * n  # equivalent charge
    
    cation_proportion = new_df.copy()

    Opct = pd.Series(index = range(0,len(new_df)),dtype='float16')
    for x in new_df.index:
        total = sum(eq_charge.iloc[x, index_first_elem:])
        Opct.iloc[x] = (O*2) / total

    for i in new_df.columns[index_first_elem:]:
        new_df[f'{i}'] = new_df[f'{i}'] * Opct # Atoms per Formula unit (APFU)
    
    if justAPFU==True:
        return new_df
    else:
        return cation_proportion, eq_charge, Opct, new_df

#####################################################################

def df_to_excel(path, new_df, shname='new sheet', mode = 'a'): 
  ''' writes the converted dataframe to a new excel sheet of a especified file or a new file
      It takes some minutes depending on the size of data, and the computer.

    mode : 'a' append a new sheet to an existing file 
           'w' creates a new file (overwrites if the file exists) on the path with the given new name 
   '''
  import pandas as pd
  with pd.ExcelWriter(path = path, mode=mode, engine="openpyxl") as writer:
      new_df.to_excel(writer,sheet_name=f'{shname}')
      writer.save()