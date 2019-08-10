#!/usr/bin/env python
# coding: utf-8

# In[1]:


import altair as alt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
os.getcwd()


# In[2]:


alt.renderers.enable('notebook')


# In[3]:


data_unemp = pd.read_excel('Unemployment.xls', sheet_name = 'Unemployment Med HH Inc')


# In[4]:


data_unemp.dropna(inplace=True)
data_unemp.reset_index(drop=True, inplace=True)
data_unemp_agg = data_unemp[data_unemp.FIPS % 1000 == 0]
data_unemp_agg.reset_index(inplace=True, drop=True)
data_unemp = data_unemp[data_unemp.FIPS % 1000 != 0]
data_unemp.head()


# In[5]:


data_unemp.info()


# In[6]:


Unemployment_rate_columns = [x for x in data_unemp.columns if 'Unemployment_rate' in x]
Unemployed_columns = [x for x in data_unemp.columns if 'Unemployed_' in x]
Employed_columns = [x for x in data_unemp.columns if 'Employed_' in x]
Civilian_labor_force_columns = [x for x in data_unemp.columns if 'Civilian_labor_force_' in x]


# In[7]:


df_map = pd.read_csv('us_map.csv')


# In[ ]:


#####Year Wise Unemployment Rate#####


# In[8]:


df_map['Unemployment_rate_2007'] = data_unemp_agg[data_unemp_agg.State.isin(df_map.code)]['Unemployment_rate_2007']
df_map['Unemployment_rate_2012'] = data_unemp_agg[data_unemp_agg.State.isin(df_map.code)]['Unemployment_rate_2012']
df_map['Unemployment_rate_2018'] = data_unemp_agg[data_unemp_agg.State.isin(df_map.code)]['Unemployment_rate_2018']

fig07 = go.Figure(data=go.Choropleth(
    locations=df_map['code'], # Spatial coordinates
    z = df_map['Unemployment_rate_2007'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'Reds',
    colorbar_title = "Unemployment Rate",
    text = df_map['state']
))

fig07.update_layout(
    title_text = '2007 US Unemployment Rate',
    geo_scope='usa', # limite map scope to USA
)

fig07.show()

fig12 = go.Figure(data=go.Choropleth(
    locations=df_map['code'], # Spatial coordinates
    z = df_map['Unemployment_rate_2012'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'Reds',
    colorbar_title = "Unemployment Rate",
    text = df_map['state']
))

fig12.update_layout(
    title_text = '2012 US Unemployment Rate',
    geo_scope='usa', # limite map scope to USA
)

fig12.show()

fig18 = go.Figure(data=go.Choropleth(
    locations=df_map['code'], # Spatial coordinates
    z = df_map['Unemployment_rate_2018'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'Reds',
    colorbar_title = "Unemployment Rate",
    text = df_map['state']
))

fig18.update_layout(
    title_text = '2018 US Unemployment Rate',
    geo_scope='usa', # limite map scope to USA
)

fig18.show()


# In[9]:


date_range = np.arange(2007,2019,1)
df_hist1 = data_unemp_agg.copy()
df_hist1['Sum_of_unemployement_rate'] = data_unemp_agg[Unemployment_rate_columns].sum(axis=1)
df_hist1_sorted = df_hist1.sort_values('Sum_of_unemployement_rate', ascending=False)


# In[ ]:


#####Unemployment Rate of States#####


# In[10]:


brush = alt.selection(type='interval', encodings=['x'])

bars = alt.Chart().mark_bar().encode(
    x='Years:Q',
    y='Unemployement Rate:Q',
    opacity=alt.condition(brush, alt.OpacityValue(1), alt.OpacityValue(0.7)),
).properties(
    width=400,
    height=300).add_selection(
    brush
)

line = alt.Chart().mark_rule(color='firebrick').encode(
    y='mean(Unemployement Rate):Q',
    size=alt.SizeValue(3)
).transform_filter(
    brush
)

df1 = df_hist1_sorted.loc[0,:][Unemployment_rate_columns].values
df1 = pd.DataFrame(df1 , date_range)
df1 = df1.reset_index()
df1.columns = ['Years','Unemployement Rate']


df2 = df_hist1_sorted.loc[1,:][Unemployment_rate_columns].values
df2 = pd.DataFrame(df2 , date_range)
df2 = df2.reset_index()
df2.columns = ['Years','Unemployement Rate']

df3 = df_hist1_sorted.loc[2,:][Unemployment_rate_columns].values
df3 = pd.DataFrame(df3 , date_range)
df3 = df3.reset_index()
df3.columns = ['Years','Unemployement Rate']

df4 = df_hist1_sorted.loc[3,:][Unemployment_rate_columns].values
df4 = pd.DataFrame(df4 , date_range)
df4 = df4.reset_index()
df4.columns = ['Years','Unemployement Rate']

df5 = df_hist1_sorted.loc[4,:][Unemployment_rate_columns].values
df5 = pd.DataFrame(df5 , date_range)
df5 = df5.reset_index()
df5.columns = ['Years','Unemployement Rate']

df6 = df_hist1_sorted.loc[5,:][Unemployment_rate_columns].values
df6 = pd.DataFrame(df6 , date_range)
df6 = df6.reset_index()
df6.columns = ['Years','Unemployement Rate']
alt.vconcat(alt.hconcat(alt.layer(bars, line, data=df1, title = '1. Unemployement Rate of %s Vs Years' %df_hist1_sorted.loc[0,'Area_name']), alt.layer(bars, line, data=df2, title = '2. Unemployement Rate of %s VsYears' %df_hist1_sorted.loc[1,'Area_name']))
,alt.hconcat(alt.layer(bars, line, data=df3, title = '3. Unemployement Rate of %s Vs Years' %df_hist1_sorted.loc[2,'Area_name']), alt.layer(bars, line, data=df4, title = '4. Unemployement Rate of %s Vs Years' %df_hist1_sorted.loc[3,'Area_name']))
,alt.hconcat(alt.layer(bars, line, data=df5, title = '5. Unemployement Rate of %s Vs Years' %df_hist1_sorted.loc[4,'Area_name']), alt.layer(bars, line, data=df6, title = '6. Unemployement Rate of %s Vs Years' %df_hist1_sorted.loc[5,'Area_name']))
)


# In[ ]:


#####Yearly Unemployment Rate w.r.t. Area#####


# In[15]:


data_unemp_agg_line = data_unemp_agg[Unemployment_rate_columns+['Area_name']].set_index('Area_name')
data_unemp_agg_line = data_unemp_agg_line.reset_index().melt('Area_name', var_name = 'Year Info', value_name='Unemployement_rate')

# source = pd.DataFrame(np.cumsum(np.random.randn(100, 3), 0).round(2),
#                     columns=['A', 'B', 'C'], index=pd.RangeIndex(100, name='x'))
# source = source.reset_index().melt('x', var_name='category', value_name='y')

# Create a selection that chooses the nearest point & selects based on x-value
nearest = alt.selection(type='single', nearest=True, on='mouseover',
                        fields=['Area_name'], empty='none')

# The basic line
line = alt.Chart(data_unemp_agg_line).mark_line(interpolate='basis').encode(
    x='Area_name:N',
    y='Unemployement_rate:Q',
    color='Year Info:N'
)
# Transparent selectors across the chart. This is what tells us
# the x-value of the cursor
selectors = alt.Chart(data_unemp_agg_line).mark_point().encode(
    x='Area_name:N',
    opacity=alt.value(0),
).add_selection(
    nearest
)

# Draw points on the line, and highlight based on selection
points = line.mark_point().encode(
    opacity=alt.condition(nearest, alt.value(1), alt.value(0))
)

# Draw text labels near the points, and highlight based on selection
text = line.mark_text(align='left', dx=5, dy=-5).encode(
    text=alt.condition(nearest, 'Unemployement_rate:Q', alt.value(' '))
)
# Draw a rule at the location of the selection
rules = alt.Chart(data_unemp_agg_line).mark_rule(color='gray').encode(
    x='Area_name:N',
).transform_filter(
    nearest
)

# Put the five layers into a chart and bind the data
alt.layer(
    line, selectors, points, rules, text, title='Yearly Unmeployement Rate w.r.t Area'
).properties(
    width=750, height=600
)


# In[16]:


data_unemp_agg_heat = data_unemp_agg.copy()
diff =  data_unemp_agg_heat['Unemployment_rate_2018'].values - data_unemp_agg_heat['Unemployment_rate_2007'].values
x, y = (list(x) for x in zip(*sorted(zip(diff, data_unemp_agg_heat['Area_name'].values), 
                                                            reverse = True)))

# Now I want to extract out the top 15 and bottom 15 countries 
Y = np.concatenate([y[0:15], y[-16:-1]])
X = np.concatenate([x[0:15], x[-16:-1]])


# In[17]:


keys = [c for c in data_unemp_agg_heat if 'rate' in c]
keys


# In[18]:


import numpy as np
import holoviews as hv
from holoviews import opts
import pandas as pd
hv.extension('bokeh')


# In[19]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


#####Movement in Unemployment rate#####


# In[20]:


country_resize = pd.melt(data_unemp_agg_heat, id_vars='Area_name',value_vars=keys, value_name='key' )
country_resize['Year'] = country_resize['variable']
country_resize['Year'] = [country_resize.loc[i,'Year'].split('_')[-1] for i in range(len(country_resize)) ]

mask = country_resize['Area_name'].isin(Y)
country_final = country_resize[mask]

# Finally plot the seaborn heatmap
plt.figure(figsize=(12,10))
country_pivot = country_final.pivot("Area_name","Year",  "key")
country_pivot = country_pivot.sort_values('2018', ascending=False)
ax = sns.heatmap(country_pivot, cmap='coolwarm', annot=False, linewidths=0, linecolor='white')
plt.title('Movement in Unemployment Rate ( Warmer: Higher rate, Cooler: Lower rate )')


# In[ ]:


#####2017 Unemployment Rate Vs Household Incomes#####


# In[23]:


source = data_unemp_agg[:20]

alt.Chart(source).mark_circle().encode(
    x='State:O',
    y='Unemployment_rate_2017:Q',
#     alt.X('Median_Household_Income_2017', scale=alt.Scale(zero=False)),
#     alt.Y('Unemployment_rate_2017', scale=alt.Scale(zero=False, padding=1)),
    color='Area_name',
    size='Median_Household_Income_2017',
	tooltip=['State', 'Unemployment_rate_2017', 'Area_name']
    
).properties(title='2017 Unemployment Rate Vs Household Incomes').interactive()


# In[ ]:


#####2017 Employment Rate Vs Household Incomes#####


# In[24]:


source = data_unemp_agg[:20]

alt.Chart(source).mark_circle().encode(
    x='State:O',
    y='Employed_2017:Q',
#     alt.X('Median_Household_Income_2017', scale=alt.Scale(zero=False)),
#     alt.Y('Unemployment_rate_2017', scale=alt.Scale(zero=False, padding=1)),
    color='Area_name',
    size='Median_Household_Income_2017',
	tooltip=['State', 'Employed_2017', 'Area_name']
    
).properties(title='2017 Employment Rate Vs Household Incomes').interactive()


# In[ ]:


#####2018 State wise Civil Force Vs Unemployed#####


# In[27]:


source = data_unemp_agg[:20]

alt.Chart(source).mark_bar(opacity=0.7).encode(
    x='State:O',
    y='Unemployed_2018:Q',
#     alt.X('Median_Household_Income_2017', scale=alt.Scale(zero=False)),
#     alt.Y('Unemployment_rate_2017', scale=alt.Scale(zero=False, padding=1)),
    color='Area_name',
    size='Civilian_labor_force_2018',
    tooltip=['State', 'Unemployed_2018', 'Civilian_labor_force_2018']
).properties(title='2018 State wise Civil Force Vs Unemployed', width = 600).interactive()


# In[ ]:


##### Net changes in Youth Unemployment for states of USA ######


# In[28]:


# Plot using Seaborn's barplot
sns.set(font_scale=1) 
f, ax = plt.subplots(figsize=(20, 25))
colors_cw = sns.color_palette('coolwarm', len(X))
sns.barplot(X, Y, palette = colors_cw[::-1])
Text = ax.set(xlabel='Decrease in Youth Unemployment Rates', 
              title='Net Increase in Youth Unemployment Rates')


# In[ ]:


##### Year wise Unemployed People Distribution #####


# In[29]:


df_Unemployed = pd.melt(data_unemp[Unemployed_columns])
df_Unemployed['Year'] = df_Unemployed['variable'].apply(lambda row: row.split('_')[1])

sns.set(rc={'figure.figsize':(15,11)})

ax1 = sns.boxplot(x="Year", y="value", data=df_Unemployed)
ax1.set(ylim=(-5, 8000))
ax1.set_title('Year wise Unemployed People Distribution')


# In[ ]:


##### Year wise Employed People Distribution #####


# In[30]:


df_Employed = pd.melt(data_unemp[Employed_columns])
df_Employed['Year'] = df_Employed['variable'].apply(lambda row: row.split('_')[1])

sns.set(rc={'figure.figsize':(15,11)})

ax2 = sns.boxplot(x="Year", y="value", data=df_Employed)
ax2.set(ylim=(-5, 80000))
ax2.set_title('Year wise Employed People Distribution')


# In[ ]:


##### Civilian labour force per year #####


# In[31]:


data_unemp['Total Civilian_labor_force'] = data_unemp[Civilian_labor_force_columns].sum(axis=1)

ax = sns.kdeplot(data_unemp[Civilian_labor_force_columns[0]], clip=(500000, 5000000) )
ax = sns.kdeplot(data_unemp[Civilian_labor_force_columns[3]], clip=(500000, 5000000) )
ax = sns.kdeplot(data_unemp[Civilian_labor_force_columns[6]], clip=(500000, 5000000) )
ax = sns.kdeplot(data_unemp[Civilian_labor_force_columns[8]], clip=(500000, 5000000) )
ax = sns.kdeplot(data_unemp[Civilian_labor_force_columns[11]], clip=(500000, 5000000) )
ax = sns.kdeplot(data_unemp['Total Civilian_labor_force'], linestyle="--", clip=(500000, 5000000) )

ax.set_title('Civilian Labor Force Comparison for 2007,2010,2013,2015,2018 & Total')


# In[ ]:


######: Median Household Income 2017 #####


# In[32]:


by = sns.distplot(data_unemp.Median_Household_Income_2017)
by.set_title('Median Household Income 2017')


# In[ ]:


##### Median Household Income of Top 5 states #####


# In[44]:


df_plot = data_unemp_agg.sort_values(["Median_Household_Income_2017"], ascending=False)
ax5 = sns.barplot(x="Area_name", y="Median_Household_Income_2017",hue="Area_name",palette="pastel", data=df_plot[:5])
ax5.set_title('Median Household Income of Top 5 states')


# In[ ]:


##### Median Household Income of Bottom 5 states #####


# In[45]:


df_plot = data_unemp_agg.sort_values(["Median_Household_Income_2017"], ascending=True)
ax6 = sns.barplot(x="Area_name", y="Median_Household_Income_2017",hue="Area_name", palette="rocket",  data=df_plot[:5])
ax6.set_title('Median Household Income of Bottom 5 states')


# In[ ]:


##### Labor Force Vs Median Income Vs Unemployed Vs Income Class 2017 #####


# In[46]:


def GenerateClass(i):
    if i>70000: #<=7:
        return 'High'
    elif (i >=60000 and i <70000): #(x[i]>7 and x[i]<=9):
        return 'Moderate'
    elif (i >=50000 and i <60000): #(x[i]>9 and x[i]<=11):
        return 'Medium'
    else:
        return 'Low'


# In[47]:


new_df = data_unemp_agg
new_df['Income tag'] = new_df['Median_Household_Income_2017'].apply(lambda row: GenerateClass(row))
new_df = new_df[['Civilian_labor_force_2017','Median_Household_Income_2017','Unemployed_2017','Income tag' ]]
new_df.head()


# In[48]:


import plotly.express as px
import plotly.graph_objects as go


# In[49]:


fig1 = px.scatter_3d(new_df, x='Civilian_labor_force_2017', y='Median_Household_Income_2017', z='Unemployed_2017',
              color='Income tag')
fig1.update_layout(
    title=go.layout.Title(
        text="Labor Force Vs Median Income Vs Unemployed Vs Income Class: 2017",
        xref="paper",
        x=0
    )
)
fig1.show()

