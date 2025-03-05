from flask import Flask
from api.vision_api import vision

app = Flask(__name__)
app.register_blueprint(vision, url_prefix='/vision')

if __name__ == '__main__':
    app.run(debug=True)