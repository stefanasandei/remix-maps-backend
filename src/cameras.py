import os
import json
import time
import math
from werkzeug.utils import secure_filename
import threading

from src.detection import detectCars

def get_cameras_metadata():
    path = "./data/cameras.json"
    if not os.path.isfile(path):
        return None
    
    camerasFile = open(path, "r")
    data = json.loads(camerasFile.read())

    camerasFile.close()

    realData = {"cameras": []}
    t = []
    for camera in data["cameras"]:
        x = threading.Thread(target=process_camera, args=(camera, realData["cameras"],))
        x.start()
        t.append(x)
    
    for thread in t:
        thread.join()

    return realData

def process_camera(camera, arr):
    if "skip" in camera:
        return
    arr.append({
        "name": camera["name"],
        "link": camera["link"],
        "processedLink": process_video_cam(camera["name"], camera["link"]),
        "cars": get_cars_cnt(camera["name"]),
        "coords": {
            "lat": camera["coords"]["lat"],
            "lon": camera["coords"]["lon"]
        }
    })

def process_video_cam(name, url):
    safe_name = secure_filename(f"{name}")
    ok = True
    if os.path.isfile(f"./data/cache/{safe_name}.json"):
        f = open(f"./data/cache/{safe_name}.json", "r")
        timestamp = json.loads(f.read())["timestamp"]

        # 1000 seconds
        force_reload = False
        if (time.time() - math.ceil(float(timestamp)) > 1000) or force_reload:
            ok = False
        else:
            ok = True
        f.close()
    else:  
        ok = False

    if not ok:
        os.system(f"curl {url} --output ./data/cache/{safe_name}.jpg")
        cnt = detectCars(f"./data/cache/{safe_name}.jpg")
        f = open(f"./data/cache/{safe_name}.json", "w")
        f.write(json.dumps({"timestamp": str(math.ceil(time.time())), "cars": cnt}))
        f.close()
    
    return f"/cache?path={safe_name}"

def get_cars_cnt(cam_name):
    safe_name = secure_filename(cam_name)
    f = open(f"./data/cache/{safe_name}.json", "r")
    data = json.loads(f.read())
    f.close()
    return data["cars"]
