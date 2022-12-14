import math
import main as lib

# download flask and create server where you can post video from phone and you get photo af the effect back.
from flask import Flask, redirect, request, session, send_file, Response, render_template
from werkzeug.utils import secure_filename

from moviepy.editor import VideoFileClip, ImageClip
import numpy as np
from PIL import Image

import zipfile
import os
from datetime import datetime
from tqdm import tqdm

HOST = '0.0.0.0'
PORT = 8000

app = Flask(__name__, static_url_path='/')

app.secret_key = 'super_secret-key!'

os.makedirs('uploads', exist_ok=True)
os.makedirs('processed', exist_ok=True)

app.config['UPLOAD_FOLDER'] = 'uploads'

app.config['MAX_CONTENT-PATH'] = 2**29

@app.get('/')
def index():
    return render_template('index.html', 
        funcs=lib.FUNC_TO_CLASS.keys(),
        func_dict=lib.FUNC_TO_CLASS
    )

@app.post('/')
def index_post():
    dt = datetime.utcnow()
    session['user_id'] = str(dt)
    key = 'timestamp'
    f = request.files['video_file']
    session['func'] = request.form['func']
    session['stretch'] = request.form['stretch'] == 'Yes' if 'stretch' in request.form else False
    session['start_time'] = request.form['start_time']
    session['length'] = request.form['length']

    filename = f'{int(dt.timestamp())}--{secure_filename(f.filename)}'
    session['filename'] = filename
    f.save(os.path.join('uploads', filename))
    return redirect(f'/processing.html?{key}={session["user_id"].replace(" ", "_")}')

def format_sse(data, event = None):
    return f'{ "event: " + event if event is not None else "" }data: {data}\n\n'

@app.get('/api/processing')
def processing():
    def process(func: str, stretch: bool, filename: str, start_time: int, length: int):
        # start SSE
        # request.headers.set('Cache-Control', 'no-store')
        # request.headers.set('Content-Type', 'text/event-stream')
        
        processing = lib.FUNC_TO_CLASS[func]

        #func = sess['func']
        file_path = os.path.join('uploads', filename)
        
        # load video file
        with VideoFileClip( file_path ) as video_file:
            # fix vertical videos
            if video_file.rotation in (90, 270):
                video_file = video_file.resize(video_file.size[::-1])
                video_file.rotation = 0

            start = start_time
            end = video_file.duration if length is None else length + start_time
            print(start, end, length)
            tt = np.arange(
                start_time,
                end,
                1.0 / video_file.fps
            )
            
            image_clip: ImageClip = video_file.to_ImageClip(0)
            img_arr = image_clip.get_frame(0)
            #result_img = eval(f'lib.{func}_start')(img_arr)
            processing.start(img_arr)
            del img_arr
            #clips_amount = len(tt)

            #step_func = eval(f'lib.{func}_step')
            l = list(enumerate(tt))
            total = len(l)
            for i, t in l: # tqdm(l, total=total, ascii=True):
                image_clip: ImageClip = video_file.to_ImageClip(t)
                img_arr = image_clip.get_frame(0)
                processing.step(img_arr)
                #result_img = step_func(result_img, img_arr)
                # send progress to client
                yield format_sse(f'{i}/{total}')
        
        #result_photo = eval(f'lib.{func}_final')(result_img, clips_amount)
        processing.end()

        if stretch:
            min_val, max_val = np.min(processing.res_photo), np.max(processing.res_photo)
            #print(f'Min: {min_val}, Max: {max_val}')
            processing.res_photo = ((processing.res_photo - min_val) * (255 / max_val)).astype(np.int64)

        result_img = processing.get_photo() #Image.fromarray( np.uint8(result_photo) )
        result_img_path = os.path.join(
            'processed',
            f'{ os.path.splitext(filename)[0].split("--")[1] }-{func}{ "-stretched" if stretch else "" }.png'
        )
        result_img.save( result_img_path )

        yield format_sse(f'{total}/{total}')

    # print(f'Session:')
    # print(session)

    length = None
    try:
        length = int(session['length'])
        if length <= 0:
            length = None
    except:
        pass
    session['result_filename'] = f'{ os.path.splitext(session["filename"])[0].split("--")[1] }-{session["func"]}{ "-stretched" if session["stretch"] else "" }.png'
    return Response(
        process(
            session['func'],
            session['stretch'],
            session['filename'],
            int(session['start_time']),
            length
        ),
        mimetype='text/event-stream'
    )
    
@app.get('/result_file')
def download():
    return send_file(
        os.path.join(
            'processed',
            session['result_filename']
        )
    )

if __name__ == '__main__':
    try:
        app.run(HOST, PORT, debug=True, threaded=True)
    finally:
        for file in os.listdir('processed'):
            os.remove(os.path.join(
                'processed',
                file
            ))
            
        uploaded_files = os.listdir('uploads')
        if len(uploaded_files) > 0:
            with zipfile.ZipFile(
                    f'uploads-{ math.floor(datetime.utcnow().timestamp())}.zip',
                    'w', zipfile.ZIP_DEFLATED
                ) as zipf:
                for file in tqdm(uploaded_files, ascii=True):
                    file_path = os.path.join(
                        'uploads',
                        file
                    )
                    zipf.write(file_path)
                    os.remove(file_path)