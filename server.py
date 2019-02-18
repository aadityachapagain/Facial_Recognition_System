import json
from time import time

from PIL import Image
from flask import Flask, request, render_template

# assuming that script is run from `server` dir
import sys, os
sys.path.append(os.path.realpath('..'))

from MTCNN import get_faces
from MTCNN import recognize, train_classifier

# For test examples acquisition
SAVE_DETECT_FILES = False
SAVE_TRAIN_FILES = False

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

print(os.getcwd())


# for CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST')  # Put any other methods you need here
    return response


@app.route('/')
def index():
    return render_template('./static/detect.html')


@app.route('/detect', methods=['POST'])
def detect():
    try:
        image_stream = request.files['image']  # get the image
        image = Image.open(image_stream)

        # Set an image confidence threshold value to limit returned data
        threshold = request.form.get('threshold')
        if threshold is None:
            threshold = 0.5
        else:
            threshold = float(threshold)

        faces = recognize(get_faces(image, threshold))

        j = json.dumps([f.data() for f in faces])
        print("Result:", j)

        # save files
        if SAVE_DETECT_FILES and len(faces):
            id = time()
            with open('test_{}.json'.format(id), 'w') as f:
                f.write(j)

            image.save('test_{}.png'.format(id))
            for i, f in enumerate(faces):
                f.img.save('face_{}_{}.png'.format(id, i))

        return j

    except Exception as e:
        import traceback
        traceback.print_exc()
        print('POST /detect error: %e' % e)
        return e


@app.route('/train', methods=['POST'])
def train():
    try:
        # image with sprites
        image_stream = request.files['image']  # get the image
        image_sprite = Image.open(image_stream)

        # forms data
        name = request.form.get('name')
        num = int(request.form.get('num'))
        size = int(request.form.get('size'))

        # save for debug purposes
        if SAVE_TRAIN_FILES:
            image_sprite.save('train_{}_{}_{}.png'.format(name, size, num))

        info = train_classifier(name, image_sprite, num, size)
        return json.dumps([{'name': n, 'train_examples': s} for n, s in info.items()])

    except Exception as e:
        import traceback
        traceback.print_exc()
        print('POST /image error: %e' % e)
        return e

app.run(debug=False, host='0.0.0.0', ssl_context='adhoc')
    # app.run(host='0.0.0.0')