# 厂区摄像头流地址与检测配置清单 (Combined)

> 整理时间: 2026-08-21  
> 数据来源: 18080 视频综合能力平台 + `工作簿1.xlsx` 检测类型配置

## 1. 代理服务启动方式

在本地启动 WebSocket → MJPEG 代理服务（OpenCV / 浏览器直接拉流）：

```bash
python ws_to_mjpeg_proxy.py --port 9000
```

---

## 2. 摄像头流地址与检测类型对照表

| 序号 | 摄像头名称 | 摄像头编码 (CameraCode) | 资源ID (ResourceId) | 通道号 | 分辨率 | 检测类型与规则 | 主码流 (1080P/高清) | 子码流 (480P/AI推荐) |
|:---:|---|---|---|:---:|:---:|---|---|---|
| **1** | 一期C2A反应釜上 | `2aafde91a91c1e1002ee2e28c66e15e7` | `2069977158886449154` | 16 | 1920x1080 | 烟、火、安全帽、涉爆车间大于3人 | `http://localhost:9000/live/2069977158886449154?stream_type=0` | `http://localhost:9000/live/2069977158886449154?stream_type=1` |
| **2** | 一期C2B反应釜 | `22dc21cecad38e0f4d7bf101c0402ff3` | `2069977158890643458` | 21 | 1920x1080 | 烟、火、安全帽、涉爆车间大于3人 | `http://localhost:9000/live/2069977158890643458?stream_type=0` | `http://localhost:9000/live/2069977158890643458?stream_type=1` |
| **3** | 一期C6反应釜 | `b30a4a8ddaa690da1c6e400ec2971d8e` | `2069977158894837762` | 17 | 1920x1080 | 烟、火、安全帽、涉爆车间大于3人 | `http://localhost:9000/live/2069977158894837762?stream_type=0` | `http://localhost:9000/live/2069977158894837762?stream_type=1` |
| **4** | 一期F1A反应釜上 | `540d499192ceadc967261fd19a3d88af` | `2069977158894837763` | 14 | 1920x1080 | 烟、火、安全帽、涉爆车间大于3人 | `http://localhost:9000/live/2069977158894837763?stream_type=0` | `http://localhost:9000/live/2069977158894837763?stream_type=1` |
| **5** | 一期F1B反应釜上 | `30b9950a2cd06abce1cae744be119bd9` | `2069977158894837764` | 15 | 1920x1080 | 烟、火、安全帽、涉爆车间大于3人 | `http://localhost:9000/live/2069977158894837764?stream_type=0` | `http://localhost:9000/live/2069977158894837764?stream_type=1` |
| **6** | 一期F3反应釜上 | `dec786bd3f01b8900e17592f6bf4de38` | `2069977158894837765` | 18 | 1920x1080 | 烟、火、安全帽、涉爆车间大于3人 | `http://localhost:9000/live/2069977158894837765?stream_type=0` | `http://localhost:9000/live/2069977158894837765?stream_type=1` |
| **7** | 一期F5反应釜上 | `d298620d68f2d590cb5fe91cace38bbb` | `2069977158899032066` | 19 | 1920x1080 | 烟、火、安全帽、涉爆车间大于3人 | `http://localhost:9000/live/2069977158899032066?stream_type=0` | `http://localhost:9000/live/2069977158899032066?stream_type=1` |
| **8** | 一期三楼储槽通道 | `5cf56d2e1d1b2b2bdabc8d5b3c6c33d9` | `2069977158899032067` | 20 | 1920x1080 | 安全帽、涉爆车间大于3人 | `http://localhost:9000/live/2069977158899032067?stream_type=0` | `http://localhost:9000/live/2069977158899032067?stream_type=1` |
| **9** | 三期侧门 | `4ea16118c0e9433ad9387f82bdddcfc6` | `2069977158899032068` | 11 | 1280x720 | *(未配置)* | `http://localhost:9000/live/2069977158899032068?stream_type=0` | `http://localhost:9000/live/2069977158899032068?stream_type=1` |
| **10** | 中控室外通道 | `10d490bedfc061dcd170f2438b9756dc` | `2069977158899032069` | 13 | 1920x1080 | *(未配置)* | `http://localhost:9000/live/2069977158899032069?stream_type=0` | `http://localhost:9000/live/2069977158899032069?stream_type=1` |
| **11** | 二期C2A反应釜 | `186e859ea7106216f396cb203dd9017b` | `2069977158903226369` | 2 | 1280x720 | 烟、火、安全帽、涉爆车间大于3人 | `http://localhost:9000/live/2069977158903226369?stream_type=0` | `http://localhost:9000/live/2069977158903226369?stream_type=1` |
| **12** | 二期C2B反应釜 | `d704bd98e5170ce37e776fe11a7ebe01` | `2069977158903226370` | 4 | 1280x720 | 烟、火、安全帽、涉爆车间大于3人 | `http://localhost:9000/live/2069977158903226370?stream_type=0` | `http://localhost:9000/live/2069977158903226370?stream_type=1` |
| **13** | 二期C2C反应釜 | `39490d2949b291228a763fa1878a8209` | `2069977158903226371` | 4 | 1280x720 | 烟、火、安全帽、涉爆车间大于3人 | `http://localhost:9000/live/2069977158903226371?stream_type=0` | `http://localhost:9000/live/2069977158903226371?stream_type=1` |
| **14** | 二期F1A反应釜 | `cee1decbc481346ed55c3624efab8b8c` | `2069977158903226372` | 0 | 2304x1296 | 烟、火、安全帽、涉爆车间大于3人 | `http://localhost:9000/live/2069977158903226372?stream_type=0` | `http://localhost:9000/live/2069977158903226372?stream_type=1` |
| **15** | 二期F1B反应釜下 | `9e8b28933e119e19dd64d4e62a463758` | `2069977158903226373` | 8 | 1920x1080 | 烟、火、安全帽、涉爆车间大于3人 | `http://localhost:9000/live/2069977158903226373?stream_type=0` | `http://localhost:9000/live/2069977158903226373?stream_type=1` |
| **16** | 二期F3B反应釜 | `1a079e161e724e191399bd41e4153573` | `2069977158907420674` | 6 | 1280x720 | 烟、火、安全帽、涉爆车间大于3人 | `http://localhost:9000/live/2069977158907420674?stream_type=0` | `http://localhost:9000/live/2069977158907420674?stream_type=1` |
| **17** | 二期F5B反应釜 | `143e98106bcec062949a07cfd67897f4` | `2069977158907420675` | 8 | 2688x1520 | 烟、火、安全帽、涉爆车间大于3人 | `http://localhost:9000/live/2069977158907420675?stream_type=0` | `http://localhost:9000/live/2069977158907420675?stream_type=1` |
| **18** | 二期二楼通道 | `3952be93bd4b3e27ff0b3f28774ac30d` | `2069977158907420676` | 10 | 1920x1080 | 烟、火、安全帽 | `http://localhost:9000/live/2069977158907420676?stream_type=0` | `http://localhost:9000/live/2069977158907420676?stream_type=1` |
| **19** | 停车场大门 | `517f9b49982ab9fc68398f0f473ce249` | `2069977158907420677` | 12 | 1920x1080 | *(未配置)* | `http://localhost:9000/live/2069977158907420677?stream_type=0` | `http://localhost:9000/live/2069977158907420677?stream_type=1` |
| **20** | 濞 | `8f5223fc6cd08c5b9750472e58c08f3a` | `2069977158907420678` | 2 | 2304x1296 | *(未配置)* | `http://localhost:9000/live/2069977158907420678?stream_type=0` | `http://localhost:9000/live/2069977158907420678?stream_type=1` |
| **21** | 通道6 | `ef251254816333345ede3544bf3d5298` | `2069977158911614978` | 6 | - | 烟、火、安全帽 | `http://localhost:9000/live/2069977158911614978?stream_type=0` | `http://localhost:9000/live/2069977158911614978?stream_type=1` |

---

## 3. 在 Python / sentry-safety 中批量导入示例

```python
import json
import cv2

# 读取结合后的配置（也可以直接读 camera_streams.json）
with open("camera_streams.json", "r", encoding="utf-8") as f:
    cameras = json.load(f)

# 示例：拉取 1 号摄像头子码流
cap = cv2.VideoCapture("http://localhost:9000/live/2069977158886449154?stream_type=1")
while True:
    ret, frame = cap.read()
    if ret:
        # 进行烟火、安全帽、人数超限等 AI 推理
        pass
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
```
