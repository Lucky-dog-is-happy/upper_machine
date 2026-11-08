# 工创赛智能物流小车上位机视觉与通信系统 (GongChuang Upper Machine)

本项目为全国大学生工程实践与创新能力大赛（工创赛）智能物流小车上位机视觉与通信核心系统。系统基于 Python 3.10+、OpenCV 与 RDLC 串口帧协议开发，支持任务码扫描、物料色块检测、地面黑环与环内数字（1/2/3）识别定位、全屏大屏状态监控以及与下位机（STM32/MCU）的实时双向闭环通信。

---

## 目录

- [一、环境准备与系统配置](#一环境准备与系统配置)
  - [1. 串口与摄像头系统权限配置](#1-串口与摄像头系统权限配置)
  - [2. 项目依赖安装与环境同步 (uv)](#2-项目依赖安装与环境同步-uv)
- [二、系统入口与多模式启动指令](#二系统入口与多模式启动指令)
  - [1. 常用模式快速指令](#1-常用模式快速指令)
  - [2. 独立调试辅助工具](#2-独立调试辅助工具)
- [三、命令行参数完整参考表](#三命令行参数完整参考表)
- [四、上下位机通信协议 (RDLC 概要)](#四上下位机通信协议-rdlc-概要)
  - [1. 链路层帧格式](#1-链路层帧格式)
  - [2. 应用层核心消息定义](#2-应用层核心消息定义)
- [五、运行时键盘交互快捷键](#五运行时键盘交互快捷键)

---

## 一、环境准备与系统配置

### 1. 串口与摄像头系统权限配置

在 Linux 系统下，串口设备（如 `/dev/ttyUSB0`）默认需要 `dialout` 用户组权限，摄像头设备需要 `video` 组权限。首次运行前请在终端执行：

```bash
# 将当前用户加入串口组 (dialout) 与摄像头组 (video)
sudo usermod -aG dialout $USER
sudo usermod -aG video $USER

# 刷新用户组生效（或注销/重启系统）
newgrp dialout
```

> **串口检测提示**：插入 USB 转串口模块后，可通过以下命令确认设备节点是否存在：
> ```bash
> ls -l /dev/ttyUSB*
> ```

---

### 2. 项目依赖安装与环境同步 (uv)

本项目推荐使用现代极速包管理器 **`uv`** 进行环境管理（项目已配置 `pyproject.toml` 与 `uv.lock`）：

```bash
# 1. 若未安装 uv，可通过官方脚本快速安装
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 在项目根目录下同步安装虚拟环境与所有依赖项
cd /home/lucky/Projects/GongChuang
uv sync
```

> **运行环境说明**：
> - 推荐直接使用 **`uv run`** 启动，无需手动切换环境；
> - 或激活本地虚拟环境后直接使用 `python`：
>   ```bash
>   source .venv/bin/activate
>   ```

---

## 二、系统入口与多模式启动指令

主运行入口统一为根目录下的 **`run_upper.py`**。

### 1. 常用模式快速指令

#### 场景 A：默认标准实机启动（比赛/全功能联调）
启用双摄像头（QR码扫描 + `/dev/video20` 物料相机）、串口通信（`/dev/ttyUSB0` 115200）以及全屏大屏 UI 状态显示：
```bash
uv run python run_upper.py
```

#### 场景 B：单摄像头物料/黑环模式（跳过二维码）
仅使用物料相机与串口（用于小车出发后阶段，或单目小车测试）：
```bash
uv run python run_upper.py --no-qr
```

#### 场景 C：纯黑环与数字识别测试（自适应小半径 r>=60）
强制初始处于圆环识别模式，自动检测最多 3 个黑环并定向识别内部数字 1、2、3：
```bash
# 桌面单目免串口快速验证
uv run python run_upper.py --mode ring --no-qr --no-serial --no-display --block-camera 0

# 实机全功能黑环测试
uv run python run_upper.py --mode ring
```

#### 场景 D：离线视觉算法调试（无串口硬件模式）
在电脑前插普通免驱 USB 摄像头进行物料/圆环识别算法调试，不连接串口硬件：
```bash
uv run python run_upper.py --no-serial --block-camera 0 --qr-camera 1
```

#### 场景 E：轻量/窗口化模式（低算力平台或开发调试）
- **窗口化运行监控屏**（不占满屏幕，方便看终端输出与 OpenCV 画面）：
  ```bash
  uv run python run_upper.py --no-fullscreen
  ```
- **完全关闭大屏 UI**（仅保留 OpenCV 画面，极简性能模式）：
  ```bash
  uv run python run_upper.py --no-display
  ```

#### 场景 F：自定义端口与参数配置
指定实际绑定的摄像头节点号、串口波特率、以及二维码广播持续时间：
```bash
uv run python run_upper.py \
  --block-camera /dev/video0 \
  --qr-camera 2 \
  --serial-port /dev/ttyUSB1 \
  --baudrate 115200 \
  --qr-send-duration 8.0
```

#### 场景 G：列出可用串口设备
```bash
uv run python run_upper.py --list-ports
```

---

### 2. 独立调试辅助工具

项目根目录下提供了针对各子功能的轻量级单模块测试工具：

1. **HSV 颜色阈值实时滑动条调参工具**：
   ```bash
   uv run python hsv_tuner.py --camera 0
   ```
2. **多目标黑环与数字融合独立测试工具**：
   ```bash
   uv run python digit_detector.py --camera 0
   ```
3. **交互式摄像头拍照采集工具**：
   ```bash
   uv run python capture_photo.py --camera 2 --prefix digits
   ```
4. **单文件版上位机运行入口**（供备用）：
   ```bash
   uv run python upper_machine.py
   ```

---

## 三、命令行参数完整参考表

| 参数名 | 默认值 | 类型 | 功能说明 |
|:---|:---:|:---:|:---|
| `--block-camera` | `/dev/video20` | str/int | 物料与黑环检测摄像头设备路径或编号 |
| `--qr-camera` | `0` | str/int | 二维码识别摄像头设备编号或路径 |
| `--no-qr` | `False` | bool | 禁用二维码相机（单摄模式：仅物料相机 + 串口） |
| `--width` / `--height` | `640` / `480` | int | 摄像头采集分辨率宽度与高度 |
| `--serial-port` | `/dev/ttyUSB0` | str | 下位机通信串口路径 |
| `--baudrate` | `115200` | int | 串口波特率 |
| `--no-serial` | `False` | bool | 禁用串口输出（无硬件调试模式） |
| `--list-ports` | - | action | 打印主机当前所有可用串口号并退出 |
| `--send-interval` | `0.5` | float | 串口向 MCU 周期性上报数据的间隔时间（秒） |
| `--qr-send-duration` | `5.0` | float | 二维码确认后广播任务码的持续秒数（0 代表仅单次立即发送） |
| `--mode` | `block` | choice | 初始识别模式：`block`（物料色块）/ `ring`（黑环与数字） |
| `--block-min-radius` | `20` | int | 物料色块最小半径（px） |
| `--block-max-radius` | `150` | int | 物料色块最大半径（px） |
| `--ring-min-radius` | `60` | int | 黑色圆环最小检测半径（px，滤除细小螺丝孔/杂点） |
| `--ring-max-radius` | `200` | int | 黑色圆环最大检测半径（px） |
| `--ring-min-area` | `100` | int | 环带内有效黑像素基准阈值（代码内自动支持周长自适应） |
| `--ring-param2` | `0.72` | float | 霍夫圆变换圆度/完美度阈值（HOUGH_GRADIENT_ALT，0.0~1.0） |
| `--digit-min-conf` | `0.65` | float | 环内数字识别置信度底线门槛（低于该值自动保底上报为 5，防止误判） |
| `--no-display` | `False` | bool | 关闭全屏任务码与机械臂抓取/放置监控屏 |
| `--no-fullscreen` | `False` | bool | 将监控屏以 1280×720 窗口化模式运行 |

---

## 四、上下位机通信协议 (RDLC 概要)

完整串口通信规范请详细参阅工程文档：[`docs/serial_protocol_upper_lower.md`](docs/serial_protocol_upper_lower.md)。

### 1. 链路层帧格式 (RDLC)
所有数据均以帧为单位传输，具备严格的头尾边界与 CRC-16/Modbus 校验：

```text
HEAD(0xC0) | SRC(1B) | DST(1B) | LEN_LO(1B) | LEN_HI(1B) | PAYLOAD(0~128B) | CRC_LO(1B) | CRC_HI(1B) | TAIL(0x0C)
```
- 上位机源地址 `0x01`，下位机目标地址 `0x02`；
- CRC-16 仅校验 `PAYLOAD` 内容。

---

### 2. 应用层核心消息定义

| 消息 ID | 标识符 | 方向 | 功能载荷格式 |
|:---:|:---|:---:|:---|
| `0x01` | **MSG_VISION** | 上位机 $\to$ 下位机 | **物料色块批量上报**（0 ~ 3 目标）：<br>`[0x01] [count(1B)] + [color_id(1B) cx(4B) cy(4B) radius(4B)] * count`<br>• count=0 表示当前无物料（载荷 `0x01 0x00`，兼通信心跳与坐标清零）<br>• color_id: 1=红, 2=黄, 3=蓝, 4=绿, 5=黑, 6=浅蓝；坐标与半径为小端浮点数 |
| `0x02` | **MSG_RING** | 上位机 $\to$ 下位机 | **黑色圆环与数字批量上报**（0 ~ 3 目标）：<br>`[0x02] [count(1B)] + [target_id(1B) cx(4B) cy(4B) radius(4B)] * count`<br>• count=0 表示当前视野无黑环（载荷 `0x02 0x00`，兼通信心跳与坐标清零）<br>• **target_id**: `1`、`2`、`3` 对应环内识别出的数字编号；`5` 为仅见黑环未看清数字 |
| `0x03` | **MSG_TASK** | 上位机 $\to$ 下位机 | **二维码任务码字符串**：<br>`[0x03] [UTF-8 string bytes...]`（如 `156+123+516+231`） |
| `0x04` | **MSG_MODE_CMD** | 下位机 $\to$ 上位机 | **下位机控制模式切换**：<br>`[0x04] [mode(1B)]`（`0x00`=识别物料，`0x01`=识别圆环） |
| `0x05` | **MSG_GRIPPER_COUNT** | 下位机 $\to$ 上位机 | **抓取与放置统计更新**：<br>`[0x05] [action(1B)]`<br>• `0x01` 抓取+1, `0x02` 抓取-1, `0x03` 放置+1, `0x04` 放置-1, `0x05` 清零<br>• `[0x05] [0x10] [pick(1B)] [place(1B)]` 绝对值精确同步 |

---

## 五、运行时键盘交互快捷键

在任意 OpenCV 图像显示窗口激活状态下，可按下述键盘按键：

- **`ESC`** 或 **`q`**：安全退出所有相机线程与程序；
- **`m`**：在 **物块检测 (BLOCK)** 与 **黑环/数字检测 (RING)** 模式间手动来回切换；
- **`p`**：在终端打印当前帧检测到的目标详细数据（坐标、半径、置信度与数字编号）。
