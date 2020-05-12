import pandas as pd
import plotly.graph_objs as go
import numpy as np
import json

# TODO: Scroll down to line 157 and set up a fifth visualization for the data dashboard

def cleandata(dataset, keepcolumns = ['Country Name', '1990', '2015'], value_variables = ['1990', '2015']):
    """Clean world bank data for a visualizaiton dashboard

    Keeps data range of dates in keep_columns variable and data for the top 10 economies
    Reorients the columns into a year, country and value
    Saves the results to a csv file

    Args:
        dataset (str): name of the csv data file

    Returns:
        None

    """    
    df = pd.read_csv(dataset, skiprows=4)

    # Keep only the columns of interest (years and country name)
    df = df[keepcolumns]

    top10country = ['United States', 'China', 'Japan', 'Germany', 'United Kingdom', 'India', 'France', 'Brazil', 'Italy', 'Canada']
    df = df[df['Country Name'].isin(top10country)]

    # melt year columns  and convert year to date time
    df_melt = df.melt(id_vars='Country Name', value_vars = value_variables)
    df_melt.columns = ['country','year', 'variable']
    df_melt['year'] = df_melt['year'].astype('datetime64[ns]').dt.year

    # output clean csv file
    return df_melt

def return_inference():
    """Creates four plotly visualizations

    Args:
        None

    Returns:
        list (dict): list containing the four plotly visualizations

    """

  # first chart plots arable land from 1990 to 2015 in top 10 economies 
  # as a line chart
    
    graph_one = []
    df = cleandata('data/API_AG.LND.ARBL.HA.PC_DS2_en_csv_v2.csv')
    df.columns = ['country','year','hectaresarablelandperperson']
    df.sort_values('hectaresarablelandperperson', ascending=False, inplace=True)
    countrylist = df.country.unique().tolist()
    
    for country in countrylist:
      x_val = df[df['country'] == country].year.tolist()
      y_val =  df[df['country'] == country].hectaresarablelandperperson.tolist()
      graph_one.append(
          go.Scatter(
          x = x_val,
          y = y_val,
          mode = 'lines',
          name = country
          )
      )

    layout_one = dict(title = 'Change in Hectares Arable Land <br> per Person 1990 to 2015',
                xaxis = dict(title = 'Year',
                  autotick=False, tick0=1990, dtick=25),
                yaxis = dict(title = 'Hectares'),
                )

# second chart plots ararble land for 2015 as a bar chart    
    graph_two = []
    df = cleandata('data/API_AG.LND.ARBL.HA.PC_DS2_en_csv_v2.csv')
    df.columns = ['country','year','hectaresarablelandperperson']
    df.sort_values('hectaresarablelandperperson', ascending=False, inplace=True)
    df = df[df['year'] == 2015] 

    graph_two.append(
      go.Bar(
      x = df.country.tolist(),
      y = df.hectaresarablelandperperson.tolist(),
      )
    )

    layout_two = dict(title = 'Hectares Arable Land per Person in 2015',
                xaxis = dict(title = 'Country',),
                yaxis = dict(title = 'Hectares per person'),
                )


# third chart plots percent of population that is rural from 1990 to 2015
    graph_three = []
    df = cleandata('data/API_SP.RUR.TOTL.ZS_DS2_en_csv_v2_9948275.csv')
    df.columns = ['country', 'year', 'percentrural']
    df.sort_values('percentrural', ascending=False, inplace=True)
    for country in countrylist:
      x_val = df[df['country'] == country].year.tolist()
      y_val =  df[df['country'] == country].percentrural.tolist()
      graph_three.append(
          go.Scatter(
          x = x_val,
          y = y_val,
          mode = 'lines',
          name = country
          )
      )

    layout_three = dict(title = 'Change in Rural Population <br> (Percent of Total Population)',
                xaxis = dict(title = 'Year',
                  autotick=False, tick0=1990, dtick=25),
                yaxis = dict(title = 'Percent'),
                )
    
# fourth chart shows rural population vs arable land
    graph_four = []
    
    valuevariables = [str(x) for x in range(1995, 2016)]
    keepcolumns = [str(x) for x in range(1995, 2016)]
    keepcolumns.insert(0, 'Country Name')

    df_one = cleandata('data/API_SP.RUR.TOTL_DS2_en_csv_v2_9914824.csv', keepcolumns, valuevariables)
    df_two = cleandata('data/API_AG.LND.FRST.K2_DS2_en_csv_v2_9910393.csv', keepcolumns, valuevariables)
    
    df_one.columns = ['country', 'year', 'variable']
    df_two.columns = ['country', 'year', 'variable']
    
    df = df_one.merge(df_two, on=['country', 'year'])

    for country in countrylist:
      x_val = df[df['country'] == country].variable_x.tolist()
      y_val = df[df['country'] == country].variable_y.tolist()
      year = df[df['country'] == country].year.tolist()
      country_label = df[df['country'] == country].country.tolist()

      text = []
      for country, year in zip(country_label, year):
          text.append(str(country) + ' ' + str(year))

      graph_four.append(
          go.Scatter(
          x = x_val,
          y = y_val,
          mode = 'markers',
          text = text,
          name = country,
          textposition = 'top right'
          )
      )

    layout_four = dict(title = 'Rural Population versus <br> Forested Area (Square Km) 1990-2015',
                xaxis = dict(title = 'Rural Population'),
                yaxis = dict(title = 'Forest Area (square km)'),
                )
    
    graph_five = []
    df_five = cleandata('data/API_SP.RUR.TOTL_DS2_en_csv_v2_9914824.csv', ['Country Name', '2015'], ['2015'])

    df_five.columns = ['country','year','ruralpopulation']
    df_five.sort_values('ruralpopulation', ascending=False, inplace=True) 

    graph_five.append(
      go.Bar(
      x = df_five.country.tolist(),
      y = df_five.ruralpopulation.tolist(),
      )
    )

    layout_five = dict(title = 'Rural Population in 2015',
                xaxis = dict(title = 'Country',),
                yaxis = dict(title = 'Rural Population'))
    
    # append all charts to the figures list
    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))
    figures.append(dict(data=graph_two, layout=layout_two))
    figures.append(dict(data=graph_three, layout=layout_three))
    figures.append(dict(data=graph_four, layout=layout_four))
    figures.append(dict(data=graph_five, layout=layout_five))

    return figures


def ml_figures():
    """Creates plotly visualizations

    Args:
        None

    Returns:
        list (dict): list containing the plotly visualizations

    """
    df = pd.read_csv('data/probabilities.csv')

    labels_path = 'data/class_index.json'
    with open(labels_path) as json_data:
        idx_to_labels = json.load(json_data)
    predicted_label = idx_to_labels[str(18)][1]

    top_probability = 0.999941

    from rsc_webapp import app
    app.logger.info(type(idx_to_labels))
    labels = []
    for k, v in idx_to_labels.items():
      labels.append(v[1])

    df = pd.read_csv('data/probabilities.csv')
    df.columns = ['class_id','probability']
    
    graph_prob = []

    graph_prob.append(
      go.Bar(
      x = df.probability.tolist(),
      y = labels,
      orientation='h',
      textposition='outside',
      marker=dict(
        color='rgba(00, 173, 255,  0.6)',
        line=dict(#007bff
            color='rgba(00, 173, 255, 1.0)',
            width=1)
        )
      )
    )
    annotations = []
    probs = np.round(df.probability.tolist(), decimals=4)

    for yd, xd in zip(probs, df.class_id.tolist()):
      # labeling bars 
      annotations.append(dict(xref='x1', yref='y1',
                              y=xd, x=yd + 3,
                              text=str(yd) + '%',
                              font=dict(family='Arial', size=12,
                                        color='rgb(96, 50, 171)'),
                              showarrow=False))

    layout_prob = dict(xaxis = dict(
                    title = 'Probability',
                    zeroline=False,
                    showline=False,
                    showticklabels=True,
                    showgrid=True,
                    domain=[0, 1]),
                yaxis = dict(dtick=1),
                height=800,
                #annotations=annotations,
                margin=dict(l=280, r=20, t=30, b=50),
                paper_bgcolor='rgb(248, 249, 250)',
                plot_bgcolor='rgb(248, 249, 250)',
                uniformtext_minsize=8, uniformtext_mode='hide'
                )
    
    # append all charts to the figures list
    figures = []
    figures.append(dict(data=graph_prob, layout=layout_prob))
    return figures, predicted_label, top_probability