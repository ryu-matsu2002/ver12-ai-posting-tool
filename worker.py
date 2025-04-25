# worker.py
from app import create_app, make_celery

flask_app = create_app()
celery    = make_celery(flask_app)
