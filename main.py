from abc import abstractmethod
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

class BaseProcessing:
    __slots__ = ('res_photo')

    def __init__(self) -> None:
        self.res_photo = None

    @abstractmethod
    def start(self, example_img: np.ndarray) -> None:
        raise NotImplementedError()

    @abstractmethod
    def step(self, new_img: np.ndarray) -> None:
        raise NotImplementedError()

    @abstractmethod
    def end(self) -> None:
        raise NotImplementedError()
    
    def get_photo(self) -> Image.Image:
        return Image.fromarray( np.uint8(self.res_photo) )

class AverageProcessing(BaseProcessing):
    __slots__ = ('amount')

    def __init__(self) -> None:
        super().__init__()
        self.amount = 0

    def start(self, example_img: np.ndarray) -> None:
        self.res_photo = np.zeros(example_img.shape, dtype=np.uint64)

    def step(self, new_img: np.ndarray) -> None:
        self.res_photo = self.res_photo + new_img.astype(dtype=np.uint64)
        self.amount += 1
        #print(self.amount)

    def end(self) -> None:
        self.res_photo = np.round( self.res_photo / self.amount ).astype(np.uint64)

class MaximumProcessing(BaseProcessing):
    def __init__(self) -> None:
        super().__init__()

    def start(self, example_img: np.ndarray):
        self.res_photo = np.zeros(example_img.shape, dtype=np.uint64)

    def step(self, new_img: np.ndarray):
        self.res_photo = np.maximum(self.res_photo, new_img)

    def end(self):
        pass

class MinimumProcessing(BaseProcessing):
    def __init__(self) -> None:
        super().__init__()

    def start(self, example_img: np.ndarray):
        self.res_photo = 255 * np.ones(example_img.shape, dtype=np.uint64)

    def step(self, new_img: np.ndarray):
        self.res_photo = np.minimum(self.res_photo, new_img)

    def end(self):
        pass

FUNC_TO_CLASS: dict[str, BaseProcessing] = {
    'avg': AverageProcessing(),
    'max': MaximumProcessing(),
    'min': MinimumProcessing()
}

# def avg_start(example_img: np.ndarray):
#     return np.zeros(example_img.shape, dtype=np.uint64)

# def avg_step(cumulative_img: np.ndarray, new_img: np.ndarray):
#     return cumulative_img + new_img

# def avg_final(cumulative_img: np.ndarray, amount: int):
#     return np.round( cumulative_img / amount )

# def max_start(example_img: np.ndarray):
#     return np.zeros(example_img.shape, dtype=np.uint64)

# def max_step(cumulative_img: np.ndarray, new_img: np.ndarray):
#     return np.maximum(cumulative_img, new_img)

# def max_final(cumulative_img: np.ndarray, _: any):
#     return cumulative_img

# def min_start(example_img: np.ndarray):
#     return 255 * np.ones(example_img.shape, dtype=np.uint64)

# def min_step(cumulative_img: np.ndarray, new_img: np.ndarray):
#     return np.minimum(cumulative_img, new_img)

# def min_final(cumulative_img: np.ndarray, _: any):
#     return cumulative_img

def main():
    args = get_args()

    processing: BaseProcessing = FUNC_TO_CLASS[args.func]

    clips_amount = None
    with VideoFileClip(args.file) as video_file:
        if video_file.rotation in (90, 270):
            video_file = video_file.resize(video_file.size[::-1])
            video_file.rotation = 0
            
        #print(video_file.size)
        height, width = video_file.size
        tt = np.arange(0, video_file.duration, 1.0 / video_file.fps)
        
        image_clip: ImageClip = video_file.to_ImageClip(0)
        
        img_arr = image_clip.get_frame(0)
        #print(img_arr.shape)
        needs_flip = list(img_arr.shape[:2]) != list(video_file.size)
        print(f'Needs flip? {needs_flip}')
        processing.start(img_arr)

        for i, t in tqdm(list(enumerate(tt)), ascii=True, dynamic_ncols=True):
            image_clip: ImageClip = video_file.to_ImageClip(t)
            img_arr: np.ndarray = image_clip.get_frame(0)
            processing.step(img_arr)
    
    processing.end()

    result_image: Image.Image = processing.get_photo()
    result_image.resize((width, height))
    result_image.save(f'{ os.path.splitext(args.file)[0] }-{args.func}.png')

if __name__ == '__main__':
    main()