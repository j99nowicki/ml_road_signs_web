from flask import Flask

app = Flask(__name__)

from rsc_webapp import routes
