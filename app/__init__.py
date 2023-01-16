import logging
from flask import Flask
from flask_login import LoginManager
from importlib import import_module


login_manager = LoginManager()
logging.basicConfig(level=logging.INFO)


def register_momo_demo(app):
    module1 = import_module('app.sample.{}.routes'.format("platelicense_demo"))
    app.register_blueprint(module1.platelicense_demo, static_folder='platelicense_demo/static')


def create_app(config):
    app = Flask(__name__)
    app.config.from_object(config)

    register_momo_demo(app)
    return app
