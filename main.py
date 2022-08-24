from argparse import ArgumentParser, Namespace
from moviepy.editor import VideoFileClip, ImageClip
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import sys

def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('-f', '--file', required=True)
    parser.add_argument('--func', default='avg', choices=['avg', 'max', 'min'])
    args = parser.parse_args()

    if os.path.splitext( args.file )[-1].lower() not in ['.mp4', '.avi', '.mov']:
        print(f'Unsupported file type: {args.file}', file=sys.stderr)
        exit(1)

    return args

def avg_start(example_img: np.ndarray):
    return np.zeros(example_img.shape, dtype=np.uint64)

def avg_step(cumulative_img: np.ndarray, new_img: np.ndarray):
    return cumulative_img + new_img

def avg_final(cumulative_img: np.ndarray, amount: int):
    return np.round( cumulative_img / amount )

def max_start(example_img: np.ndarray):
    return np.zeros(example_img.shape, dtype=np.uint64)

def max_step(cumulative_img: np.ndarray, new_img: np.ndarray):
    return np.maximum(cumulative_img, new_img)

def max_final(cumulative_img: np.ndarray, _: any):
    return cumulative_img

def min_start(example_img: np.ndarray):
    return 255 * np.ones(example_img.shape, dtype=np.uint64)

def min_step(cumulative_img: np.ndarray, new_img: np.ndarray):
    return np.minimum(cumulative_img, new_img)

def min_final(cumulative_img: np.ndarray, _: any):
    return cumulative_img

def main():
    args = get_args()

    clips_amount = None
    with VideoFileClip(args.file) as video_file:
        tt = np.arange(0, video_file.duration, 1.0 / video_file.fps)
        
        image_clip: ImageClip = video_file.to_ImageClip(0)
        img_arr = image_clip.get_frame(0)
        result_img = eval(f'{args.func}_start')(img_arr)
        clips_amount = len(tt)

        dims = reversed(result_img.shape[:2])
        print(f'Dimensions of result image: {dims[0]}x{dims[1]}')
        step_func = eval(f'{args.func}_step')
        for i, t in tqdm(list(enumerate(tt)), ascii=True, dynamic_ncols=True):
            image_clip: ImageClip = video_file.to_ImageClip(t)
            img_arr = image_clip.get_frame(0)
            result_img = step_func(result_img, img_arr)
    
    result_photo = eval(f'{args.func}_final')(result_img, clips_amount)

    # width, height, color_depth = result_photo.shape
    result_img = Image.fromarray( np.uint8(result_photo) )
    result_img.save(f'{ os.path.splitext(args.file)[0] }-{args.func}.png')

if __name__ == '__main__':
    main()