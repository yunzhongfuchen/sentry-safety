"""BM1684X 多路视频解码 + 多模型队列式并发推理压测

参考 benchmark_multi_model_parallel.py 架构：
- 每路视频独立硬解线程，每秒收集最新帧
- 模型分 N 个队列，队列内串行，队列间并行
- 支持 batch 推理（若 bmodel 的 batch_size > 1）

Usage (on BM1684X device):
    python3 benchmark_sophon.py \
        --video ./datasets/test_car_person_1080P.mp4 \
        --bmodel ./models/BM1684X/yolov8s_fp16_1b.bmodel \
        --streams 4 \
        --num-models 6 \
        --num-queues 2 \
        --duration 30 \
        --infer-interval 1.0
"""
import os
import sys
import time
import json
import argparse
import threading
import signal
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

import sophon.sail as sail

# ---------- postprocess (inlined) ----------
class _pseudo_torch_nms:
    def nms_boxes(self, boxes, scores, iou_thres):
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        areas = w * h
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])
            w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
            h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
            inter = w1 * h1
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_thres)[0]
            order = order[inds + 1]
        return np.array(keep)

    def xywh2xyxy(self, x):
        y = x.copy() if isinstance(x, np.ndarray) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y

    def non_max_suppression(
        self, prediction, conf_thres=0.25, iou_thres=0.5,
        classes=None, agnostic=False, multi_label=False,
        labels=(), max_det=300, nm=0
    ):
        bs = prediction.shape[0]
        nc = prediction.shape[2] - nm - 4
        mi = 4 + nc
        xc = prediction[:, :, 4:mi].max(2) > conf_thres
        max_wh = 7680
        max_nms = 30000
        multi_label &= nc > 1
        output = [np.zeros((0, 6 + nm))] * bs
        for xi, x in enumerate(prediction):
            x = x[xc[xi]]
            if not x.shape[0]:
                continue
            box, cls, mask = x[:, :4], x[:, 4:nc + 4], x[:, nc + 4:]
            box = self.xywh2xyxy(box)
            if multi_label:
                i, j = (cls > conf_thres).nonzero()
                x = np.concatenate(
                    (box[i], x[i, 4 + j, None], j[:, None].astype(np.float32), mask[i]), 1
                )
            else:
                conf = cls.max(1, keepdims=True)
                j_argmax = cls.argmax(1)
                j = j_argmax if j_argmax.shape == x[:, 5:].shape else np.expand_dims(j_argmax, 1)
                x = np.concatenate((box, conf, j.astype(np.float32), mask), 1)[conf.reshape(-1) > conf_thres]
            n = x.shape[0]
            if not n:
                continue
            x_argsort = np.argsort(x[:, 4])[::-1][:max_nms]
            x = x[x_argsort]
            c = x[:, 5:6] * (0 if agnostic else max_wh)
            boxes, scores = x[:, :4] + c, x[:, 4]
            i = self.nms_boxes(boxes, scores, iou_thres)
            if i.shape[0] > max_det:
                i = i[:max_det]
            output[xi] = x[i]
        return output


class _PostProcess:
    def __init__(self, conf_thresh=0.001, nms_thresh=0.7, agnostic=False, multi_label=True, max_det=300):
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.agnostic_nms = agnostic
        self.multi_label = multi_label
        self.max_det = max_det
        self.nms = _pseudo_torch_nms()

    def __call__(self, preds_batch, org_size_batch, ratios_batch, txy_batch):
        if isinstance(preds_batch, list) and len(preds_batch) == 1:
            dets = np.concatenate(preds_batch)
        else:
            raise NotImplementedError
        outs = self.nms.non_max_suppression(
            dets, self.conf_thresh, self.nms_thresh,
            agnostic=False, max_det=300, multi_label=self.multi_label, classes=None
        )
        for det, (org_w, org_h), ratio, (tx1, ty1) in zip(
            outs, org_size_batch, ratios_batch, txy_batch
        ):
            if len(det):
                coords = det[:, :4]
                coords[:, [0, 2]] -= tx1
                coords[:, [1, 3]] -= ty1
                coords[:, [0, 2]] /= ratio[0]
                coords[:, [1, 3]] /= ratio[1]
                coords[:, [0, 2]] = coords[:, [0, 2]].clip(0, org_w - 1)
                coords[:, [1, 3]] = coords[:, [1, 3]].clip(0, org_h - 1)
                det[:, :4] = coords
        return outs


PostProcess = _PostProcess
# ---------- end postprocess ----------


class SharedPreprocessor:
    """共享预处理器：所有模型共用同一套预处理结果"""

    def __init__(self, dev_id: int, input_shape, input_dtype, input_scale, img_dtype):
        self.dev_id = dev_id
        self.handle = sail.Handle(dev_id)
        self.bmcv = sail.Bmcv(self.handle)
        self.input_shape = input_shape
        self.input_dtype = input_dtype
        self.img_dtype = img_dtype
        self.net_h = input_shape[2]
        self.net_w = input_shape[3]
        self.ab = [x * input_scale / 255.0 for x in [1, 0, 1, 0, 1, 0]]
        self.preprocess_time = 0.0
        self._lock = threading.Lock()

    def preprocess_batch(self, bmimg_list: List[sail.BMImage]):
        """预处理一批帧，返回 (tensor_or_list, meta_list)"""
        if not bmimg_list:
            return None, []

        t0 = time.perf_counter()
        n = len(bmimg_list)
        preprocessed_list = []
        meta_list = []

        for bmimg in bmimg_list:
            rgb_planar = sail.BMImage(
                self.handle, bmimg.height(), bmimg.width(),
                sail.Format.FORMAT_RGB_PLANAR, sail.DATA_TYPE_EXT_1N_BYTE,
            )
            self.bmcv.convert_format(bmimg, rgb_planar)

            img_w = rgb_planar.width()
            img_h = rgb_planar.height()
            r = min(self.net_w / img_w, self.net_h / img_h)
            tw = int(round(r * img_w))
            th = int(round(r * img_h))
            tx1 = (self.net_w - tw) / 2
            ty1 = (self.net_h - th) / 2

            attr = sail.PaddingAtrr()
            attr.set_stx(int(round(tx1 - 0.1)))
            attr.set_sty(int(round(ty1 - 0.1)))
            attr.set_w(tw)
            attr.set_h(th)
            attr.set_r(114)
            attr.set_g(114)
            attr.set_b(114)

            resized = self.bmcv.crop_and_resize_padding(
                rgb_planar, 0, 0, img_w, img_h,
                self.net_w, self.net_h, attr,
                sail.bmcv_resize_algorithm.BMCV_INTER_LINEAR,
            )

            preprocessed = sail.BMImage(
                self.handle, self.net_h, self.net_w,
                sail.Format.FORMAT_RGB_PLANAR, self.img_dtype,
            )
            self.bmcv.convert_to(
                resized, preprocessed,
                ((self.ab[0], self.ab[1]), (self.ab[2], self.ab[3]), (self.ab[4], self.ab[5])),
            )

            preprocessed_list.append(preprocessed)
            meta_list.append((img_w, img_h, r, tx1, ty1))

        batch_size = self.input_shape[0]
        if batch_size == 1:
            # 1b 模型：逐帧生成 tensor list
            tensor_list = []
            for p in preprocessed_list:
                t = sail.Tensor(self.handle, self.input_shape, self.input_dtype, False, False)
                self.bmcv.bm_image_to_tensor(p, t)
                tensor_list.append(t)
            result = tensor_list
        else:
            # 4b/多 batch 模型：按 batch_size 分组
            tensor_list = []
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                b = end - start
                input_tensor = sail.Tensor(self.handle, self.input_shape, self.input_dtype, False, False)
                if b == 1:
                    self.bmcv.bm_image_to_tensor(preprocessed_list[start], input_tensor)
                else:
                    BMImageArray = getattr(sail, f'BMImageArray{batch_size}D')
                    bmimgs = BMImageArray()
                    for i in range(b):
                        bmimgs[i] = preprocessed_list[start + i].data()
                    self.bmcv.bm_image_to_tensor(bmimgs, input_tensor)
                tensor_list.append(input_tensor)
            result = tensor_list

        t1 = time.perf_counter()
        with self._lock:
            self.preprocess_time += (t1 - t0) * 1000

        return result, meta_list


class SophonEngine:
    """单个 bmodel 推理引擎（只做 inference + postprocess，预处理由 SharedPreprocessor 统一做）"""

    def __init__(self, bmodel_path: str, dev_id: int, conf_thresh=0.25, nms_thresh=0.7):
        self.dev_id = dev_id
        self.handle = sail.Handle(dev_id)

        self.net = sail.Engine(bmodel_path, dev_id, sail.IOMode.SYSO)
        self.graph_name = self.net.get_graph_names()[0]

        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.input_dtype = self.net.get_input_dtype(self.graph_name, self.input_name)
        self.input_scale = self.net.get_input_scale(self.graph_name, self.input_name)
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name)
        self.input_shapes = {self.input_name: self.input_shape}
        self.batch_size = self.input_shape[0]

        self.output_names = self.net.get_output_names(self.graph_name)
        self.output_tensors = {}
        for name in self.output_names:
            shape = self.net.get_output_shape(self.graph_name, name)
            dtype = self.net.get_output_dtype(self.graph_name, name)
            self.output_tensors[name] = sail.Tensor(self.handle, shape, dtype, True, True)

        self.postprocess = PostProcess(
            conf_thresh=conf_thresh, nms_thresh=nms_thresh,
            agnostic=False, multi_label=False, max_det=300,
        )

        self.inference_time = 0.0
        self.postprocess_time = 0.0
        self.infer_count = 0
        self._lock = threading.Lock()

    def infer_from_tensor(self, input_tensor: sail.Tensor, meta_list: list, n: int):
        """从已预处理的 tensor 做推理，返回 (results, infer_ms, post_ms)"""
        t0 = time.perf_counter()

        input_tensors = {self.input_name: input_tensor}
        self.net.process(self.graph_name, input_tensors, self.input_shapes, self.output_tensors)

        t1 = time.perf_counter()

        outputs_dict = {}
        for name in self.output_names:
            outputs_dict[name] = self.output_tensors[name].asnumpy()[:n]

        out_keys = list(outputs_dict.keys())
        ord_idx = []
        for name in self.output_names:
            for i, k in enumerate(out_keys):
                if name in k:
                    ord_idx.append(i)
                    break
        preds = [outputs_dict[out_keys[i]] for i in ord_idx]

        ori_sizes = [(m[0], m[1]) for m in meta_list]
        ratios = [(m[2], m[2]) for m in meta_list]
        txys = [(m[3], m[4]) for m in meta_list]
        results = self.postprocess(preds, ori_sizes, ratios, txys)

        t2 = time.perf_counter()

        infer_ms = (t1 - t0) * 1000
        post_ms = (t2 - t1) * 1000
        with self._lock:
            self.inference_time += infer_ms
            self.postprocess_time += post_ms
            self.infer_count += 1

        return results[:n], infer_ms, post_ms

    def report(self):
        with self._lock:
            n = self.infer_count or 1
            return {
                "inference_ms": self.inference_time / n,
                "postprocess_ms": self.postprocess_time / n,
                "total_ms": (self.inference_time + self.postprocess_time) / n,
                "infer_count": self.infer_count,
            }


class StreamDecoder:
    """单路视频硬解线程"""

    _init_lock = threading.Lock()

    def __init__(self, video_path: str, stream_id: int, dev_id: int, sample_fps: float = 0):
        self.video_path = video_path
        self.stream_id = stream_id
        self.dev_id = dev_id
        self.sample_fps = sample_fps
        self.decoder = None
        self.handle = None
        self.running = False
        self.thread = None
        self.frame_count = 0
        self.infer_ready_count = 0
        self.latest_bmimg = None
        self._lock = threading.Lock()
        self._start_time = 0.0

    def start(self) -> bool:
        with StreamDecoder._init_lock:
            self.decoder = sail.Decoder(str(self.video_path), True, self.dev_id)
        if not self.decoder.is_opened():
            print(f"[stream-{self.stream_id}] decoder open failed")
            return False
        self.handle = sail.Handle(self.dev_id)
        self.running = True
        self._start_time = time.perf_counter()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return True

    def _loop(self):
        target_interval = 1.0 / self.sample_fps if self.sample_fps > 0 else 1.0 / 30.0
        while self.running:
            t0 = time.perf_counter()
            bmimg = sail.BMImage()
            ret = self.decoder.read(self.handle, bmimg)
            if ret != 0:
                # loop video
                with StreamDecoder._init_lock:
                    self.decoder.release()
                    self.decoder = sail.Decoder(str(self.video_path), True, self.dev_id)
                continue

            self.frame_count += 1
            with self._lock:
                self.latest_bmimg = bmimg

            # 帧率限速
            elapsed = time.perf_counter() - t0
            sleep_time = target_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def snapshot(self):
        with self._lock:
            return self.latest_bmimg

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.decoder:
            self.decoder.release()

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._start_time) * 1000


class ModelQueue(threading.Thread):
    """模型队列：内部串行执行多个 SophonEngine，与其他队列并行"""

    def __init__(self, queue_id: int, engines: List[SophonEngine]):
        super().__init__(daemon=True)
        self.queue_id = queue_id
        self.engines = engines
        self.input_tensor = None
        self.meta_list = []
        self.n = 0
        self.start_event = threading.Event()
        self.done_event = threading.Event()
        self.running = True
        self.infer_count = 0
        self.total_infer_time = 0.0
        self.last_round_ms = 0.0
        self.last_round_engine_stats = []  # [(infer_ms, post_ms, img_n), ...]
        self._lock = threading.Lock()

    def run(self):
        while self.running:
            self.start_event.wait()
            self.start_event.clear()
            if not self.running:
                break

            t0 = time.perf_counter()
            tensor = self.input_tensor
            meta = self.meta_list
            n = self.n

            round_stats = []
            # 队列内串行：每个 engine 依次处理同一批预处理结果
            for engine in self.engines:
                if tensor is not None:
                    if isinstance(tensor, list):
                        # tensor list：每个元素是一个 batch tensor
                        idx = 0
                        total_infer = 0.0
                        total_post = 0.0
                        total_n = 0
                        for batch_tensor in tensor:
                            batch_n = min(engine.batch_size, self.n - idx)
                            batch_meta = meta[idx:idx + batch_n]
                            _, infer_ms, post_ms = engine.infer_from_tensor(
                                batch_tensor, batch_meta, len(batch_meta)
                            )
                            total_infer += infer_ms
                            total_post += post_ms
                            total_n += len(batch_meta)
                            idx += len(batch_meta)
                        round_stats.append((total_infer, total_post, total_n))
                    else:
                        _, infer_ms, post_ms = engine.infer_from_tensor(tensor, meta, n)
                        round_stats.append((infer_ms, post_ms, n))

            round_ms = (time.perf_counter() - t0) * 1000
            with self._lock:
                self.infer_count += 1
                self.total_infer_time += round_ms
                self.last_round_ms = round_ms
                self.last_round_engine_stats = round_stats

            self.done_event.set()

    def set_tensor(self, tensor, meta_list, n):
        self.input_tensor = tensor
        self.meta_list = meta_list
        self.n = n
        self.start_event.set()

    def stop(self):
        self.running = False
        self.start_event.set()
        self.join(timeout=5.0)

    @property
    def avg_queue_ms(self) -> float:
        with self._lock:
            return (self.total_infer_time / self.infer_count) if self.infer_count else 0.0


OUTPUT_DIR = Path(__file__).parent / "test_results" / "benchmark_sophon"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--bmodel", type=str, required=True)
    parser.add_argument("--dev_id", type=int, default=0)
    parser.add_argument("--streams", type=int, default=4, help="视频路数")
    parser.add_argument("--num-models", type=int, default=6, help="模型副本总数")
    parser.add_argument("--num-queues", type=int, default=2, help="并行队列数（<= num-models）")
    parser.add_argument("--duration", type=int, default=30, help="压测时长(秒)")
    parser.add_argument("--sample-fps", type=float, default=0, help="采样帧率，0=全部")
    parser.add_argument("--infer-interval", type=float, default=1.0, help="推理调度间隔(秒)")
    parser.add_argument("--interval", type=int, default=5, help="状态打印间隔(秒)")
    parser.add_argument("--conf_thresh", type=float, default=0.25)
    parser.add_argument("--nms_thresh", type=float, default=0.7)
    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"Video not found: {args.video}")
        sys.exit(1)
    if not Path(args.bmodel).exists():
        print(f"Bmodel not found: {args.bmodel}")
        sys.exit(1)
    if args.num_queues < 1:
        args.num_queues = 1
    if args.num_queues > args.num_models:
        args.num_queues = args.num_models

    # 视频信息
    cap = cv2.VideoCapture(args.video)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    video_duration = total_frames / video_fps if total_frames > 0 else 0

    sample_interval = 0
    if args.sample_fps > 0:
        sample_interval = max(1, int(round(video_fps / args.sample_fps)))

    print(f"\n{'='*60}")
    print(f"BM1684X 队列式压测")
    print(f"{'='*60}")
    print(f"  视频: {args.video} ({video_w}x{video_h}, {video_fps:.1f}fps)")
    print(f"  模型: {args.bmodel}")
    print(f"  路数: {args.streams}")
    print(f"  模型副本: {args.num_models}")
    print(f"  队列数: {args.num_queues} (队列内串行，队列间并行)")
    print(f"  压测时长: {args.duration}s")
    print(f"  调度间隔: {args.infer_interval}s")
    if sample_interval:
        print(f"  采样: 每{sample_interval}帧取1帧")
    else:
        print(f"  采样: 全部")
    print(f"{'='*60}\n")

    # 加载模型
    print(f"Loading {args.num_models} model copies...")
    engines = []
    for i in range(args.num_models):
        t0 = time.perf_counter()
        engine = SophonEngine(args.bmodel, args.dev_id, args.conf_thresh, args.nms_thresh)
        t1 = time.perf_counter()
        engines.append(engine)
        print(f"  Model {i} loaded in {(t1-t0)*1000:.1f} ms, batch_size={engine.batch_size}")

    # 共享预处理器（所有模型共用）
    first_engine = engines[0]
    preprocessor = SharedPreprocessor(
        args.dev_id,
        first_engine.input_shape,
        first_engine.input_dtype,
        first_engine.input_scale,
        first_engine.img_dtype if hasattr(first_engine, 'img_dtype') else sail.DATA_TYPE_EXT_FLOAT32,
    )
    print(f"  Shared preprocessor created\n")

    # 分配队列
    queue_size = (args.num_models + args.num_queues - 1) // args.num_queues
    workers: List[ModelQueue] = []
    for q in range(args.num_queues):
        start = q * queue_size
        end = min(start + queue_size, args.num_models)
        q_engines = engines[start:end]
        w = ModelQueue(q, q_engines)
        w.start()
        workers.append(w)
        print(f"  Queue{q}: models {start}-{end-1} ({len(q_engines)} engines)")
    print()

    # 启动解码流
    stop_event = threading.Event()

    def on_signal(*_):
        print("\n收到中断，正在停止...")
        stop_event.set()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    decoders: List[StreamDecoder] = []
    for i in range(args.streams):
        d = StreamDecoder(args.video, i, args.dev_id, args.sample_fps)
        if d.start():
            decoders.append(d)
            print(f"[stream-{i}] started")
        else:
            print(f"[stream-{i}] start failed, skip")
            break
        time.sleep(0.2)

    if not decoders:
        print("No stream started")
        sys.exit(1)

    # 压测主循环
    print(f"\nStarting benchmark for {args.duration}s...")
    start = time.perf_counter()
    last_print = start
    infer_count = 0
    total_frames_inferred = 0

    while not stop_event.is_set():
        t0 = time.perf_counter()

        # 收集各路最新帧
        frames = []
        for d in decoders:
            f = d.snapshot()
            if f is not None:
                frames.append(f)

        if frames:
            # 预处理只做一次
            input_tensor, meta_list = preprocessor.preprocess_batch(frames)
            n = len(frames)

            # 所有队列共享同一预处理结果
            for w in workers:
                w.set_tensor(input_tensor, meta_list, n)

            for w in workers:
                w.done_event.wait()
                w.done_event.clear()

            infer_count += 1
            total_frames_inferred += n * args.num_models

        # 状态打印
        now = time.perf_counter()
        if now - last_print >= args.interval:
            elapsed = now - start
            total_decoded = sum(d.frame_count for d in decoders)
            print(f"\n--- [{elapsed:5.1f}s] 汇总 ---")
            print(f"  解码帧={total_decoded}, 解码FPS={total_decoded/elapsed:.1f}, "
                  f"推理次数={infer_count}, 推理帧={total_frames_inferred}")
            for w in workers:
                print(f"  Queue{w.queue_id}: {w.last_round_ms:.1f} ms/轮")
            print(f"  瓶颈队列: {max(w.last_round_ms for w in workers):.1f} ms")
            print("  【队列耗时】")
            for w in workers:
                q_ms = w.last_round_ms
                print(f"    队列{w.queue_id}: 单轮={q_ms:.1f}ms")
                print("    【模型明细】")
                for idx, (infer_ms, post_ms, img_n) in enumerate(w.last_round_engine_stats):
                    per_infer = infer_ms / img_n if img_n else 0.0
                    per_post = post_ms / img_n if img_n else 0.0
                    print(f"      模型{idx}:")
                    print(f"        单次 inference ={per_infer:.1f}ms, "
                          f"单次 postprocess={per_post:.1f}ms")
                    print(f"        总 inference   ={infer_ms:.1f}ms, "
                          f"总 postprocess  ={post_ms:.1f}ms, "
                          f"总耗时={infer_ms + post_ms:.1f}ms "
                          f"(处理{img_n}张)")
            last_print = now

        if now - start >= args.duration:
            print(f"\n达到时长 {args.duration}s，停止...")
            break

        sleep_time = args.infer_interval - (time.perf_counter() - t0)
        if sleep_time > 0:
            time.sleep(sleep_time)

    # 最终报告
    print("\n========== 最终报告 ==========")
    for w in workers:
        w.stop()
    for d in decoders:
        d.stop()

    elapsed = time.perf_counter() - start
    total_decoded = sum(d.frame_count for d in decoders)

    print(f"压测时长:      {elapsed:.1f}s")
    print(f"并发路数:      {len(decoders)}")
    print(f"模型副本:      {args.num_models}")
    print(f"队列数:        {args.num_queues}")
    print(f"总解码帧数:    {total_decoded}")
    print(f"总解码 FPS:    {total_decoded / elapsed:.2f}")
    print(f"推理次数:      {infer_count}")
    print(f"总推理帧数:    {total_frames_inferred}")
    print()

    print("\n【队列耗时】")
    queue_avgs = []
    for w in workers:
        q_ms = w.avg_queue_ms
        queue_avgs.append(q_ms)
        q_total = w.total_infer_time
        print(f"  队列{w.queue_id}: 单轮={q_ms:.1f}ms, 总耗时={q_total:.1f}ms, 轮数={w.infer_count}")
        print("  【模型明细】")
        for idx, e in enumerate(w.engines):
            r = e.report()
            total_infer = r['inference_ms'] * r['infer_count']
            total_post = r['postprocess_ms'] * r['infer_count']
            total_model = total_infer + total_post
            print(f"    模型{idx}:")
            print(f"      单次 inference ={r['inference_ms']:.1f}ms, "
                  f"单次 postprocess={r['postprocess_ms']:.1f}ms")
            print(f"      总 inference   ={total_infer:.1f}ms, "
                  f"总 postprocess  ={total_post:.1f}ms, "
                  f"总耗时={total_model:.1f}ms "
                  f"(调用{r['infer_count']}次)")
    print(f"\n共享预处理:    {preprocessor.preprocess_time / infer_count:.1f} ms/轮" if infer_count else "")
    print(f"瓶颈队列:      {max(queue_avgs):.1f} ms/轮")
    print(f"理论吞吐量:    {1000 / max(queue_avgs) * len(frames) if max(queue_avgs) > 0 else 'N/A'} 帧/秒")
    print("==============================")

    # 保存报告
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("BM1684X 队列式压测报告\n")
        f.write("=" * 40 + "\n")
        f.write(f"压测时长: {elapsed:.1f}s\n")
        f.write(f"路数: {len(decoders)}\n")
        f.write(f"模型: {args.num_models}\n")
        f.write(f"队列: {args.num_queues}\n")
        f.write(f"解码帧: {total_decoded}\n")
        f.write(f"解码FPS: {total_decoded / elapsed:.2f}\n")
        f.write(f"推理帧: {total_frames_inferred}\n")
        for w in workers:
            f.write(f"队列{w.queue_id}: {w.avg_queue_ms:.1f} ms\n")
    print(f"\n报告已保存: {report_path}")


if __name__ == "__main__":
    main()
