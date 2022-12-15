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
    parser.add_argument('--func', default='avg', choices=FUNC_TO_CLASS.keys())
    parser.add_argument('-s', '--stretch', default=False, const=True, nargs='?')

    args = parser.parse_args()

    if os.path.splitext( args.file )[-1].lower() not in ['.mp4', '.avi', '.mov']:
        print(f'Unsupported file type: {args.file}', file=sys.stderr)
        exit(1)

    return args

class BaseProcessing:
    __slots__ = ('res_photo', 'name')

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
        self.name = 'Average'
        self.amount = 0

    def start(self, example_img: np.ndarray) -> None:
        self.res_photo = np.zeros(example_img.shape, dtype=np.uint64)

    def step(self, new_img: np.ndarray) -> None:
        self.res_photo = self.res_photo + new_img.astype(dtype=np.uint64)
        self.amount += 1

    def end(self) -> None:
        print(f'Frames processed: { self.amount }')
        self.res_photo = np.round( self.res_photo / self.amount ).astype(np.uint64)

class AdditiveProcessing(BaseProcessing):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'Additive'

    def start(self, example_img: np.ndarray) -> None:
        self.res_photo = np.zeros(example_img.shape, dtype=np.uint64)

    def step(self, new_img: np.ndarray) -> None:
        self.res_photo = self.res_photo + new_img.astype(dtype=np.uint64)

    def end(self) -> None:
        division_rate = (np.max(self.res_photo) // 255) + 1
        print(f'Division rate: {division_rate}')
        self.res_photo = np.round( self.res_photo / division_rate ).astype(np.uint64)

class MaximumProcessing(BaseProcessing):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'Maximum'

    def start(self, example_img: np.ndarray):
        self.res_photo = np.zeros(example_img.shape, dtype=np.uint64)

    def step(self, new_img: np.ndarray):
        self.res_photo = np.maximum(self.res_photo, new_img)

    def end(self):
        pass

class MinimumProcessing(BaseProcessing):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'Minimum'

    def start(self, example_img: np.ndarray):
        self.res_photo = 255 * np.ones(example_img.shape, dtype=np.uint64)

    def step(self, new_img: np.ndarray):
        self.res_photo = np.minimum(self.res_photo, new_img)

    def end(self):
        pass


FUNC_TO_CLASS: dict[str, BaseProcessing] = {
    'avg': AverageProcessing(),
    'max': MaximumProcessing(),
    'min': MinimumProcessing(),
    'add': AdditiveProcessing()
}

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
        #print(f'Needs flip? {needs_flip}')
        processing.start(img_arr)

        for i, t in tqdm(list(enumerate(tt)), ascii=True, dynamic_ncols=True):
            image_clip: ImageClip = video_file.to_ImageClip(t)
            img_arr: np.ndarray = image_clip.get_frame(0)
            processing.step(img_arr)
    
    processing.end()

    if args.stretch:
        min_val, max_val = np.min(processing.res_photo), np.max(processing.res_photo)
        print(f'Min: {min_val}, Max: {max_val}')
        processing.res_photo = ((processing.res_photo - min_val) * (255 / max_val)).astype(np.int64)

    result_image: Image.Image = processing.get_photo()

    result_image.resize((width, height))
    result_image.save(f'{ os.path.splitext(args.file)[0] }-{args.func}{ "-stretch" if args.stretch else "" }.png')

if __name__ == '__main__':
    main()