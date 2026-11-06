# 上下位机串口通信协议与启动说明 (RDLC)

> **简介概括**  
> 本文档说明工创赛上位机与下位机（MCU）的串口通信协议与启动参数。通信基于 RDLC 帧协议与 CRC-16 校验，波特率 115200。支持传输二维码任务码、物料色块与黑色圆环坐标（单帧最多上报 3 个目标，含加权融合中心点与数字编号），并支持下位机指令切换物料与圆环检测模式。系统入口为 run_upper.py，默认启用 /dev/ttyUSB0 串口与双摄像头；支持通过 --no-qr 切换单摄模式，并可配置二维码广播持续时间。

---

## 一、系统启动参数说明

系统主入口脚本为 `run_upper.py`（底层解耦模块位于 `upper_machine/`）。

### 1. 默认参数表

| 参数名 | 默认值 | 类型 | 功能说明 |
|---|---|---|---|
| `--serial-port` | `/dev/ttyUSB0` | str | 串口设备路径，默认启动即自动连接 |
| `--baudrate` | `115200` | int | 串口通信波特率 |
| `--no-serial` | `False` | bool | 禁用串口连接（单机无硬件调试时使用） |
| `--block-camera` | `/dev/video20` | str/int | 物块与圆环检测摄像头，支持路径或编号 |
| `--qr-camera` | `0` | str/int | 二维码识别摄像头 |
| `--no-qr` | `False` | bool | 单摄模式：关闭二维码相机，仅运行物料/圆环相机与串口 |
| `--qr-send-duration` | `5.0` | float | 二维码确认后持续广播秒数，超时后自动停发任务码 |
| `--send-interval` | `0.5` | float | 串口上报数据的时间间隔（秒） |
| `--mode` | `block` | choice | 初始检测模式（`block` 物料 / `ring` 圆环） |
| `--list-ports` | - | action | 列出主机所有可用串口设备并退出 |

### 2. 常用启动命令

```bash
# 1. 默认最简启动（启用串口 /dev/ttyUSB0 + 双摄 QR 0 与物料 /dev/video20）
python run_upper.py

# 2. 单摄像头模式（跳过二维码，仅使用物块相机 /dev/video20 与串口）
python run_upper.py --no-qr

# 3. 调整任务码广播持续时间（例如确认后广播 8 秒）
python run_upper.py --qr-send-duration 8.0

# 4. 离线视觉调试（不打开串口，仅在屏幕显示识别画面）
python run_upper.py --no-serial

# 5. 独立调试数字 1、2、3 与圆环加权融合算法（单模块测试，不影响主流程）
python digit_detector.py
```

---

## 二、链路层帧结构 (RDLC)

上位机与下位机之间所有串口数据均封装为 RDLC 帧，带头尾校验：

```text
HEAD(0xC0) | SRC(1B) | DST(1B) | LEN_LO(1B) | LEN_HI(1B) | PAYLOAD(0~128B) | CRC_LO(1B) | CRC_HI(1B) | TAIL(0x0C)
```

- **帧起始 (HEAD)**: 固定 `0xC0`
- **源地址 (SRC)**: 上位机 `0x01`，下位机 `0x02`
- **目的地址 (DST)**: 上位机 `0x01`，下位机 `0x02`
- **载荷长度 (LEN)**: 2 字节小端序，表示 PAYLOAD 的字节数
- **校验码 (CRC)**: 2 字节 CRC-16/Modbus（仅校验 PAYLOAD 内容，小端序）
- **帧结束 (TAIL)**: 固定 `0x0C`

---

## 三、应用层协议 (PAYLOAD)

PAYLOAD 的首字节为 **消息类型 ID**：

| ID | 名称 | 方向 | 说明 |
|---|---|---|---|
| `0x01` | MSG_VISION | 上位机 → 下位机 | 视觉物料色块坐标批量上报 |
| `0x02` | MSG_RING | 上位机 → 下位机 | 地面圆环 / 融合中心坐标批量上报 |
| `0x03` | MSG_TASK | 上位机 → 下位机 | 二维码任务码字符串 |
| `0x04` | MSG_MODE_CMD | 下位机 → 上位机 | 检测模式切换指令（物料 / 圆环） |
| `0x05` | MSG_GRIPPER_COUNT | 下位机 → 上位机 | 抓取与放置统计更新（加减计数或同步） |

---

### 1. MSG_VISION (0x01) — 物料色块检测

用于批量上报视野内的圆形色块像素坐标。

```text
[msg_id(1B)] [count(1B)] + [color_id(1B) cx(4B) cy(4B) radius(4B)] * count
```

- `msg_id`: 固定 `0x01`
- `count`: 有效色块数量（`1 ~ 3`）
- 单个物料信息（固定 13 字节）：
  - `color_id` (uint8): 颜色编号（1=红, 2=黄, 3=蓝, 4=绿, 5=黑, 6=浅蓝）
  - `cx` (float32): 质心 X 坐标（小端序浮点数）
  - `cy` (float32): 质心 Y 坐标（小端序浮点数）
  - `radius` (float32): 外接圆半径（小端序浮点数）

---

### 2. MSG_RING (0x02) — 圆环与数字识别

用于批量上报地面黑色圆环或圆环-数字加权融合坐标，单帧支持 1~3 个目标。

```text
[msg_id(1B)] [count(1B)] + [target_id(1B) cx(4B) cy(4B) radius(4B)] * count
```

- `msg_id`: 固定 `0x02`
- `count`: 目标数量（`1 ~ 3`）
- 单个目标信息（固定 13 字节）：
  - `target_id` (uint8):
    - `1`, `2`, `3`: 识别到圆环中心对应的数字编号（1号环、2号环、3号环）
    - `5`: 仅识别到黑色圆环，未看清数字（对应颜色表中 5=黑）
  - `cx` (float32): 加权融合或圆环中心的 X 坐标（小端序）
  - `cy` (float32): 加权融合或圆环中心的 Y 坐标（小端序）
  - `radius` (float32): 圆环半径（小端序）

#### 动态数量与多目标机制说明
- 若视野中有 3 个目标：`count = 3`，单帧载荷 41 字节，一次性送达。
- 若视野中仅有 2 个目标（如仅看到 1、2 号）：`count = 2`，载荷 28 字节。
- 若 3 号环数字被遮挡：上报 3 个目标，其中 1、2 号为数字编号，3 号上报 `target_id = 5` 保底提供坐标。

---

### 3. MSG_TASK (0x03) — 二维码任务码

传输二维码解析后的任务编码字符串（格式如 `156+123+516+231`）。

```text
[msg_id(1B)] [task_code_bytes(NB)]
```
- `msg_id`: 固定 `0x03`
- `task_code_bytes`: UTF-8 编码字符串，长度由 RDLC 帧长自动决定。
- 任务码广播机制：二维码连续匹配 3 次确认后，上位机开始以 `--send-interval` 频率持续广播，经过 `--qr-send-duration`（默认 5.0 秒）后彻底停止，随后仅上报物料/圆环数据。

---

### 4. MSG_MODE_CMD (0x04) — 模式切换指令

下位机发送给上位机，控制上位机工作模式。

```text
[msg_id(1B)] [mode(1B)]
```
- `msg_id`: 固定 `0x04`
- `mode`: 
  - `0x00`: 切换为物料识别模式（上位机回复 `MSG_VISION`）
  - `0x01`: 切换为黑色圆环识别模式（上位机回复 `MSG_RING`）

---

### 5. MSG_GRIPPER_COUNT (0x05) — 抓取与放置统计指令

下位机发送给上位机，用于动态加减或同步全屏显示器底部的抓取与放置成功次数。

```text
[msg_id(1B)] [action(1B)] [optional data...]
```
- `msg_id`: 固定 `0x05`
- `action`:
  - `0x01`: 抓取成功次数 +1
  - `0x02`: 抓取成功次数 -1
  - `0x03`: 放置成功次数 +1
  - `0x04`: 放置成功次数 -1
  - `0x05`: 计数值清零（抓取=0，放置=0）
  - `0x10`: 直接同步精确计数值：`[0x05] [0x10] [pick(1B)] [place(1B)]`

#### 下位机发送示例
- 机械臂抓取完成，发送 2 字节：`05 01`（上位机屏幕“抓取”次数 +1）
- 机械臂放置完成，发送 2 字节：`05 03`（上位机屏幕“放置”次数 +1）
- 绝对同步数值（例如抓取 3 次、放置 2 次），发送 4 字节：`05 10 03 02`

---

## 四、下位机 C 语言解析与发送示例

```c
void parse_payload(uint8_t* payload, uint16_t len) {
    if (len < 1) return;
    uint8_t msg_id = payload[0];

    switch (msg_id) {
        case 0x01:   // MSG_VISION (物料)
        case 0x02: { // MSG_RING (圆环/数字)
            if (len < 2) return;
            uint8_t count = payload[1];
            if (count > 3) count = 3;

            for (uint8_t i = 0; i < count; i++) {
                uint16_t offset = 2 + i * 13;
                if (offset + 13 > len) break;

                uint8_t id = payload[offset];
                float cx = *(float*)&payload[offset + 1];
                float cy = *(float*)&payload[offset + 5];
                float r  = *(float*)&payload[offset + 9];

                if (msg_id == 0x01) {
                    process_material_block(id, cx, cy, r);
                } else {
                    process_ring_target(id, cx, cy, r);
                }
            }
            break;
        }

        case 0x03: { // MSG_TASK (任务码)
            char task_code[32] = {0};
            uint16_t copy_len = (len - 1 < 31) ? len - 1 : 31;
            memcpy(task_code, &payload[1], copy_len);
            process_task_code(task_code);
            break;
        }

        default:
            break;
    }
}

// 下位机上报抓取/放置计数的发送封装示例
void report_gripper_event(uint8_t action) {
    // action: 0x01(抓取+1), 0x02(抓取-1), 0x03(放置+1), 0x04(放置-1), 0x05(清零)
    uint8_t payload[2] = {0x05, action};
    rdlc_send(payload, 2); 
}

void report_gripper_sync(uint8_t pick_count, uint8_t place_count) {
    uint8_t payload[4] = {0x05, 0x10, pick_count, place_count};
    rdlc_send(payload, 4);
}
```
