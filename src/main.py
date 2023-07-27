from src.cameras import get_cameras_metadata
from flask import Flask, request, send_file
import json
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/")
@cross_origin()
def ruok():
    return "imok"

@app.route("/cameras")
@cross_origin()
def cameras():
    metadata = get_cameras_metadata()
    return metadata

@app.route("/frame")
@cross_origin()
def camera_info():
    id = request.args.get('name')

    metadata = get_cameras_metadata()
    data = json.loads(metadata)

    found = -1
    for index, camera in enumerate(data["cameras"]):
        if camera["name"] == id:
            found = index
            break

    if found == -1:
        return {}

    return data["cameras"][found]["link"]

@app.route("/cache")
@cross_origin()
def get_cached_img():
    path = request.args.get("path")
    return send_file(f"../data/cache/{path}.jpg", mimetype='image/gif')
