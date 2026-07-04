#!/usr/bin/env python3
"""Benchmark YOLOv8 BMCV inference on 32 images."""
import os
import time
import argparse
import glob
import numpy as np
import sophon.sail as sail
from sophon_demo_release.sample.YOLOv8_plus_det.python.yolov8_bmcv import YOLOv8, draw_bmcv
from sophon_demo_release.sample.YOLOv8_plus_det.python.utils import COCO_CLASSES, COLORS


def build_file_list(input_path, max_images=32):
    """Collect up to max_images image files from input_path."""
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')
    files = []
    if os.path.isdir(input_path):
        for ext in exts:
            files.extend(glob.glob(os.path.join(input_path, ext)))
            files.extend(glob.glob(os.path.join(input_path, '**', ext), recursive=True))
    elif os.path.isfile(input_path):
        files = [input_path]
    else:
        raise FileNotFoundError(f'input not found: {input_path}')
    files = sorted(list(set(files)))[:max_images]
    return files


def benchmark(args):
    if not os.path.exists(args.bmodel):
        raise FileNotFoundError(f'bmodel not found: {args.bmodel}')

    file_list = build_file_list(args.input, args.count)
    if len(file_list) < args.count:
        print(f'Warning: only found {len(file_list)} images, padding by repeating last image.')
        while len(file_list) < args.count:
            file_list.append(file_list[-1])

    yolov8 = YOLOv8(args)
    batch_size = yolov8.batch_size
    handle = sail.Handle(args.dev_id)

    # Decode all images once (not timed)
    bmimg_list_all = []
    for f in file_list:
        decoder = sail.Decoder(f, True, args.dev_id)
        bmimg = sail.BMImage()
        ret = decoder.read(handle, bmimg)
        if ret != 0:
            raise RuntimeError(f'decode failed: {f}')
        bmimg_list_all.append(bmimg)

    # Warm-up
    print('Warming up ...')
    for _ in range(args.warmup):
        _ = yolov8(bmimg_list_all[:batch_size])
    yolov8.init()

    # Benchmark
    print(f'Benchmarking {args.count} images, batch_size={batch_size}, loops={args.loops} ...')
    total_times = []
    decode_times = []
    preprocess_times = []
    inference_times = []
    postprocess_times = []

    for loop in range(args.loops):
        yolov8.init()
        loop_decode = 0.0

        start_total = time.time()
        for i in range(0, args.count, batch_size):
            batch = bmimg_list_all[i:i + batch_size]
            # Pad batch if needed
            while len(batch) < batch_size:
                batch.append(batch[-1])

            # We reuse decoded images; to include decode time we re-decode one image per slot
            # But since images are already decoded, we simulate by not counting decode in this benchmark
            # If you want strict decode timing, use original script. Here we measure pipeline time.
            _ = yolov8(batch)

        end_total = time.time()
        total_times.append(end_total - start_total)
        preprocess_times.append(yolov8.preprocess_time)
        inference_times.append(yolov8.inference_time)
        postprocess_times.append(yolov8.postprocess_time)

    # Report
    def stats(arr):
        arr_ms = np.array(arr) * 1000
        return arr_ms.mean(), arr_ms.std()

    total_mean, total_std = stats(total_times)
    pre_mean, pre_std = stats(preprocess_times)
    inf_mean, inf_std = stats(inference_times)
    post_mean, post_std = stats(postprocess_times)

    print('\n' + '=' * 60)
    print(f'Benchmark Result: {args.count} images x {args.loops} loops')
    print(f'  Batch size     : {batch_size}')
    print(f'  Total time     : {total_mean:.2f} ± {total_std:.2f} ms  (per loop)')
    print(f'  Preprocess     : {pre_mean:.2f} ± {pre_std:.2f} ms')
    print(f'  Inference      : {inf_mean:.2f} ± {inf_std:.2f} ms')
    print(f'  Postprocess    : {post_mean:.2f} ± {post_std:.2f} ms')
    print(f'  FPS            : {args.count / (total_mean / 1000):.2f}')
    print(f'  Avg per image  : {total_mean / args.count:.2f} ms')
    print('=' * 60)


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default='sophon-demo-release/sample/YOLOv8_plus_det/datasets/test',
                        help='path of input image or directory')
    parser.add_argument('--bmodel', type=str,
                        default='sophon-demo-release/sample/YOLOv8_plus_det/models/BM1684X/yolov8s_fp32_1b.bmodel',
                        help='path of bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    parser.add_argument('--conf_thresh', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--nms_thresh', type=float, default=0.7, help='nms threshold')
    parser.add_argument('--count', type=int, default=32, help='number of images to process')
    parser.add_argument('--loops', type=int, default=5, help='number of benchmark loops')
    parser.add_argument('--warmup', type=int, default=3, help='warmup loops')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argsparser()
    benchmark(args)
