# -*- encoding: utf-8 -*-
import os
from decouple import AutoConfig
from ctypes import cdll, c_char_p

config = AutoConfig(search_path=os.getcwd()+"/env")


class Config(object):
    basedir = os.path.abspath(os.path.dirname(__file__))

    # Set up the App SECRET_KEY
    SECRET_KEY = config('SECRET_KEY', default='S#perS3crEt_007')

    # This will create a file in <app> FOLDER
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'db.sqlite3')
    SQLALCHEMY_TRACK_MODIFICATIONS = False


class DebugConfig(Config):
    DEBUG = True

    # lib = cdll.LoadLibrary("./app/passwd.so")
    # lib.passedDecrypt.argtypes = [c_char_p]
    # lib.passedDecrypt.restype = c_char_p

    # enc_pass = config('DB_PASS', default='1234567890')
    # password = lib.passedDecrypt(enc_pass.encode()).decode()

    # # Security
    # SESSION_COOKIE_HTTPONLY = True
    # REMEMBER_COOKIE_HTTPONLY = True
    # REMEMBER_COOKIE_DURATION = 3600

    # SQLALCHEMY_ENGINE_OPTIONS = {
    #     'pool_pre_ping': True,
    #     'pool_recycle': 7200,
    #     'pool_timeout': 100,
    #     'pool_size': 300,
    #     'max_overflow': 10,
    # }

    # SQLALCHEMY_DATABASE_URI = '{}://{}:{}@{}:{}/{}'.format(
    #     config('DB_ENGINE', default='mysql+pymysql'),
    #     config('DB_USERNAME', default='test'),
    #     password,
    #     config('DB_HOST', default='127.0.0.1'),
    #     config('DB_PORT', default=3306),
    #     config('DB_NAME', default='testdb')
    # )


# Load all possible configurations
config_dict = {
    'Debug': DebugConfig
}
