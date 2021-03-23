import os

from flask import Flask

def create_app(test_config=None):
    app = Flask(__name__,instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        # DATABASE = os.path.join(app.instance_path,'flaskr.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    from . import forcast_15
    app.register_blueprint(forcast_15.bp)

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host="0.0.0.0", port="5000", processes=8, threaded=False)
 