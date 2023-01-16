import os
from flask import Blueprint

URI = "demo"

platelicense_demo = Blueprint(
    'momo_blueprint',
    __name__,
    url_prefix=f'/{URI}/platelicense_demo/',
    template_folder='templates',
    static_folder=''
)

