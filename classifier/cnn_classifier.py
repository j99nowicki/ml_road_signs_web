from __future__ import print_function
from rsc_webapp import app
import torch
import pandas as pd
import plotly.graph_objs as go
import numpy as np
import json
from torchvision import transforms
from torch.nn import functional as F
from PIL import Image
import time
import os
import random
import string
from classifier.model_1 import visualize_stn
import matplotlib.pyplot as plt

#import torch
#from torchvision import datasets, models, transforms

def generate_filename(size=10, chars=string.ascii_uppercase + string.digits, extension='png'):
  """Creates random filename

    Args:
        size: length of the filename part, without dot and extention
        chars: character range to draw random characters from
        extension: extension to be added to the returned filenam

    Returns:
        random filame with extension

    """
  filename = ''.join(random.choice(chars) for _ in range(size))
  return filename + '.' + extension
  

def ml_figures(input_filename):
    """ 
    Performs inference on the input image and returns data 
    to be visualized by the web page

    Args:
        input_filename: full path to the input file

    Returns:
        figures: dict list containing plotly graph with probabilities for all classes
        predicted_label: a string with predicted label name
        iconpath: a path to predicted sign icon
        maxConfidenceValue_str: confidence value for the top prediction
        eval_time_str: time it took to load and evaluate the model
        filename_stn_in: path to STN input file 
        filename_stn_out: path to STN output file

    """

    model = app.config['MODEL']
    transform_evaluate = app.config['TRANSFORM_EVALUATE'] 

    img_paths = [input_filename]
    img_list = [Image.open(img_path) for img_path in img_paths]
    start_time = time.perf_counter()
    input_batch = torch.stack([transform_evaluate(img).to('cpu') for img in img_list])
    pred_tensor = model(input_batch)
    pred_probs = F.softmax(pred_tensor, dim=1).cpu().data.numpy()
    end_time = time.perf_counter()
    eval_time_str = "{:.4f}".format(end_time - start_time)
#    app.logger.info("evaluation time: {} seconds".format(eval_time_str))
    maxConfidenceValue = np.amax(pred_probs[0,:])
    maxConfidenceValue_str = "{:.4f}".format(maxConfidenceValue)
    maxConfidenceClass = np.where(pred_probs[0,:] == maxConfidenceValue)[0][0]
#    app.logger.info('maxConfidenceClass: {}'.format(maxConfidenceClass))
#    app.logger.info('maxConfidenceValue: {}'.format(maxConfidenceValue_str))
    
    # STN Visualizations
    data = torch.stack([transform_evaluate(img).to('cpu') for img in img_list])
    input_grid, transformed_grid = visualize_stn( model, data)
    filename_stn_in = os.path.join(app.config['UPLOAD_FOLDER'], generate_filename(10))
    filename_stn_out = os.path.join(app.config['UPLOAD_FOLDER'], generate_filename(10))
    plt.imsave(filename_stn_in, input_grid, cmap='Greys')
    plt.imsave(filename_stn_out, transformed_grid, cmap='Greys')

    iconpath = app.config['ICONS_FOLDER'] + '/'+str(maxConfidenceClass)+".png"

    idx_to_labels = app.config['IDX_TO_LABELS']
    labels = app.config['LABELS']

    predicted_label = idx_to_labels[str(maxConfidenceClass)][1]
    graph_prob = []

    graph_prob.append(
      go.Bar(
      x = pred_probs[0],
      y = labels,
      orientation='h',
      showlegend=False,
      textposition='outside',
      marker=dict(
        color='rgba(23, 162, 184,  0.6)', ##17a2b8
        line=dict(
            color='rgba(23, 162, 184, 1.0)',
            width=1)
        )
      )
    )
    '''annotations = []
    probs = np.round(df.probability.tolist(), decimals=4)
    for yd, xd in zip(probs, df.class_id.tolist()):
      # labeling bars 
      annotations.append(dict(xref='x1', yref='y1',
                              y=xd, x=yd + 3,
                              text=str(yd) + '%',
                              font=dict(family='Arial', size=12,
                                        color='rgb(96, 50, 171)'),
                              showarrow=False))
    '''

    layout_prob = dict(xaxis = dict(
                    title = 'Probability',
                    zeroline=False,
                    showline=False,
                    showticklabels=True,
                    showgrid=True,
                    domain=[0, 1],
                    autorange=False, 
                    range=[0, 1],
                    tick=0.1),
                yaxis = dict(dtick=1),
                height=900,
                #annotations=annotations,
                margin=dict(l=300, r=20, t=30, b=50),
                paper_bgcolor='rgb(248, 249, 250)',
                plot_bgcolor='rgb(248, 249, 250)',
                uniformtext=dict(minsize=9, mode='hide')               
                )
    
    # append all charts to the figures list
    figures = []
    figures.append(dict(data=graph_prob, layout=layout_prob))
    return figures, predicted_label, iconpath, maxConfidenceValue_str, eval_time_str, filename_stn_in, filename_stn_out