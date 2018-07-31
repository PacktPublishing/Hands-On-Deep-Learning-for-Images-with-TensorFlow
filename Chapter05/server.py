"""
This is the server module to create a `connexion` application. Run it from the
command line with python for testing.abs

The application is named to be compatible with uwsgi.
"""

import os

import connexion

PORT = int(os.environ.get('PORT', 5000))

application = connexion.App(__name__, port=PORT, specification_dir='')
application.add_api('models.yaml')


if __name__ == '__main__':
    application.run(server='tornado', debug=True)
