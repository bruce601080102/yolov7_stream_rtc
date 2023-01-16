import requests
from flask import render_template, request, Response
from app.sample.platelicense_demo import platelicense_demo


@platelicense_demo.route("/", methods=["GET"])
def index():
    return render_template('index.html', name='home')


