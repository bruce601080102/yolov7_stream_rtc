# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""
import os
from flask_migrate import Migrate
from flask_cors import CORS
from app import create_app
from config import config_dict


# The configuration
get_config_mode = 'Debug'
try:
    # Load the configuration using the default values
    app_config = config_dict[get_config_mode.capitalize()]
except KeyError:
    exit('Error: Invalid <config_mode>. Expected values [Debug, Test, Production]')


app = create_app(app_config)
# Migrate(app, db)
CORS(app)


@app.route('/')
def hello():
    return 'Hello World!'


# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=80, use_reloader=True)
