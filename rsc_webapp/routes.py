from rsc_webapp import app
import json, plotly
from flask import render_template, request, redirect
from wrangling_scripts.wrangle_data import return_figures
import os

@app.route('/')
@app.route('/index')
def index():

    figures = return_figures()

    # plot ids for the html id tag
    ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]

    # Convert the plotly figures to JSON for javascript in html template
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('index.html',
                           ids=ids,
                           figuresJSON=figuresJSON)

app.config["IMAGE_UPLOADS"] = "static/img/uploads"

@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():

    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.file_name)
            print("Image saves: " + image.file_name)
            return redirect(request.url)        
    return render_template('templates/upload_image.html')
