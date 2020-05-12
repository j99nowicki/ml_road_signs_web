from __future__ import print_function
import torch
import pandas as pd
import plotly.graph_objs as go
import numpy as np
import json
#import torch
#from torchvision import datasets, models, transforms



def return_inference(img_path=None):
    """Creates four plotly visualizations

    Args:
        img_path: path to input image

    Returns:
        list (dict): list with predictions for each category

    """
    '''
    model.eval()
    img_transform=data_transform_test(Image.open(img_path)).to(device)
    input = torch.stack([img_transform])

    output = model(input)
    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)
    
    #pred_label_idx.squeeze_()
    #visualize_stn(input)
    
    return pred_label_idx
    '''
    return


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
    return figures, predicted_label, top_probability, torch.rand(5, 3)