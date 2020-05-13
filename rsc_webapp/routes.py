from rsc_webapp import app
import json, plotly
from flask import render_template, request, redirect, send_from_directory, url_for
from wrangling_scripts.wrangle_data import return_figures
from classifier.cnn_classifier import return_inference, ml_figures
from classifier.model_1 import BaselineNet
import os
import logging
from werkzeug.utils import secure_filename
from werkzeug.exceptions import HTTPException
import urllib.request
import tarfile
import time
import torch
from torchvision import transforms


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['MYDIR'] = os.path.dirname(__file__)
app.config['UPLOAD_FOLDER_REL'] = '/static/img/uploads'
app.config['IMG_FOLDER_REL'] = '/static/img'
app.config['ICONS_FOLDER'] = '/static/img/icons'
app.config['UPLOAD_FOLDER'] = app.config['MYDIR'] + app.config['UPLOAD_FOLDER_REL']
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 # file size limit: 5MB
app.config['INITIAL_SIGN'] = app.config['MYDIR'] + app.config['IMG_FOLDER_REL']+ '/attention_sign.png'

logging.basicConfig(level=logging.DEBUG)

def extract_tar_gz(filename, destination_dir):
    with tarfile.open(filename, 'r:gz') as _tar:
        _tar.extractall(destination_dir)

@app.before_first_request
def initalize():
    app.logger.info('Initalizing a model')
    urllib.request.urlretrieve ('https://rsc-public-static.s3.amazonaws.com/model-pt/model-1-99-177.tar.gz', app.config['MYDIR']  +'/static/ml/model-1-99-177.tar.gz')
    extract_tar_gz(app.config['MYDIR'] +'/static/ml/model-1-99-177.tar.gz', app.config['MYDIR'] +'/static/ml/')
    model = BaselineNet().to('cpu')
    model.load_state_dict(torch.load(app.config['MYDIR'] +'/static/ml/model.pre-trained_5', map_location=torch.device('cpu')))
    model.eval()
    transform_evaluate = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    app.config['MODEL'] = model
    app.config['TRANSFORM_EVALUATE'] = transform_evaluate

    app.logger.info('Loading labels')
    labels_path = 'data/class_index.json'
    with open(labels_path) as json_data:
        idx_to_labels = json.load(json_data)
    app.config['IDX_TO_LABELS'] = idx_to_labels

    labels = []
    for k, v in idx_to_labels.items():
      labels.append(v[1])
    app.config['LABELS'] = labels

    #make sure the directory for uploading files is created
    try:
        os.mkdir(app.config['UPLOAD_FOLDER'])
    except FileExistsError:
        pass

    return

@app.route('/')
@app.route('/index')
def index(filename=None):

    #remove older files than 5 min from /upload
    now = time.time()
    for f in os.listdir(app.config['UPLOAD_FOLDER']):
        if os.stat(os.path.join(app.config['UPLOAD_FOLDER'],f)).st_mtime < now - 1*60*5:
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'],f))

    if filename==None:
        filename = app.config['INITIAL_SIGN']
        render_filename = filename[len(app.config['MYDIR']):] 
    else:
        render_filename = filename[len(app.config['UPLOAD_FOLDER']) - len(app.config['UPLOAD_FOLDER_REL']):] 

    app.logger.info("filename {}".format(filename))

    figures, sign_name, iconpath, top_probability, eval_time_str, filename_stn_in, filename_stn_out= ml_figures(filename)

    # plot ids for the html id tag
    ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]

    # Convert the plotly figures to JSON for javascript in html template
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)
    
    render_filename_stn_in = filename_stn_in[len(app.config['UPLOAD_FOLDER']) - len(app.config['UPLOAD_FOLDER_REL']):] 
    render_filename_stn_out = filename_stn_out[len(app.config['UPLOAD_FOLDER']) - len(app.config['UPLOAD_FOLDER_REL']):] 

    return render_template('index.html',
                            ids=ids,
                            figuresJSON=figuresJSON,
                            iconpath=iconpath,
                            input_filename=render_filename,
                            sign_name=sign_name, 
                            probability=str(top_probability), 
                            eval_time=eval_time_str, 
                            filename_stn_in=render_filename_stn_in,
                            filename_stn_out=render_filename_stn_out)

@app.route('/figures')
def figures():
    figures = return_figures()

    # plot ids for the html id tag
    ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]

    # Convert the plotly figures to JSON for javascript in html template
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('index.html',
                           ids=ids,
                           figuresJSON=figuresJSON)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload-image', methods=['GET', 'POST'])
def upload_image():
    app.logger.info("/upload-image method:{}, request.files: {}".format(request.method,request.files ))

    if request.method == "POST":
        # check if the post request has the file part
        if 'image' not in request.files:
            app.logger.info('No image uploaded')
            return redirect(request.url)
        file = request.files['image']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            app.logger.info('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filename_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filename_path)
            #return redirect(url_for('uploaded_file',filename=filename))
            return index(filename_path)
    return render_template('upload_image.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


@app.errorhandler(413)
def forbidden(e):
    app.logger.info(str(e.code) + ": " + e.name + ". " + e.description)
    return render_template("error.html"), 413

'''
@app.errorhandler(HTTPException)
def handle_exception(e):
    """Return custom HTTP error page."""
    # start with the correct headers and status code from the error
    response = e.get_response()
    error_message = ""
    if e.code==413:
        error_message = "Maximum allowd size is 5MB"
    app.logger.info(str(e.code) + ": " + e.name + ". " + e.description)
    return render_template('error.html',
                           error_code=str(e.code) + ": " + e.name + " " + e.description,
                           error_message=error_message), e.code

@app.errorhandler(HTTPException)
def handle_exception(e):
    """Return JSON instead of HTML for HTTP errors."""
    # start with the correct headers and status code from the error
    response = e.get_response()
    # replace the body with JSON
    response.data = json.dumps({
        "code": e.code,
        "name": e.name,
        "description": e.description,
    })
    response.content_type = "application/json"
    return response
'''