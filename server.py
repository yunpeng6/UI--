# from flask import Flask
# from api.vision_api import vision
#
# app = Flask(__name__)
# app.register_blueprint(vision, url_prefix='/vision')
#
# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask
from flask import jsonify
from flask_cors import CORS
from api.vision_api import vision

app = Flask(__name__)
CORS(app)

app.register_blueprint(vision, url_prefix='/vision')


@app.errorhandler(Exception)
def error(e):
   ret = dict()
   ret["code"] = 1
   ret["data"] = repr(e)
   return jsonify(ret)


if __name__ == '__main__':
   app.run(host="0.0.0.0", port=9092)