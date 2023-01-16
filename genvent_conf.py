import gevent.monkey
gevent.monkey.patch_all()

bind = "0.0.0.0:9997"

workers = 1
worker_class = 'gevent'
threads = 3
