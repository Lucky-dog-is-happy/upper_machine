// IMU 里程计教程 —— 启动方法与原理
// 编译：typst compile imu_odom_tutorial.typ
#set page(
  paper: "a4",
  margin: (top: 2.4cm, bottom: 2.2cm, x: 2.1cm),
  numbering: "- 1 -",
  header: context {
    let p = counter(page).get().first()
    if p > 1 {
      text(8.5pt, fill: luma(110))[IMU 里程计教程 #h(1fr) 192.168.3.243 · ROS2 Humble]
      v(-0.4em)
      line(length: 100%, stroke: 0.5pt + luma(180))
    }
  },
)
#set text(font: ("Noto Sans CJK SC", "Source Han Sans", "Noto Serif CJK SC"), size: 10.5pt, lang: "zh")
#set par(justify: true, leading: 0.85em, spacing: 0.75em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): set text(size: 16pt)
#show heading.where(level: 2): set text(size: 13pt)
#show heading.where(level: 3): set text(size: 11.5pt)
#show raw: set text(font: ("Maple Mono NF", "Noto Sans Mono CJK SC", "DejaVu Sans Mono"), size: 8.6pt)
#show raw.where(block: true): it => block(
  fill: luma(245),
  stroke: 0.5pt + luma(200),
  inset: 9pt,
  radius: 3pt,
  width: 100%,
  breakable: true,
  it,
)
#show link: set text(fill: rgb("#1a5fb4"))
#show figure.caption: set text(size: 9pt)
#show table.cell: set par(justify: false)
#set figure(gap: 0.6em)

// 便签框
#let note(title, body) = block(
  fill: rgb("#eef4fb"),
  stroke: 1pt + rgb("#8fb6e8"),
  radius: 4pt,
  inset: 10pt,
  width: 100%,
)[
  #text(weight: "bold", fill: rgb("#1a5fb4"))[#title]
  #v(0.3em)
  #body
]

#let warn(title, body) = block(
  fill: rgb("#fdf1e7"),
  stroke: 1pt + rgb("#f0a868"),
  radius: 4pt,
  inset: 10pt,
  width: 100%,
)[
  #text(weight: "bold", fill: rgb("#b3540a"))[#title]
  #v(0.3em)
  #body
]

#let tip(title, body) = block(
  fill: rgb("#ecf7ee"),
  stroke: 1pt + rgb("#7ec98f"),
  radius: 4pt,
  inset: 10pt,
  width: 100%,
)[
  #text(weight: "bold", fill: rgb("#1e7a37"))[#title]
  #v(0.3em)
  #body
]

// ── 封面 ──────────────────────────────────────────────────────────────
#page(header: none)[
  #v(3.2cm)
  #align(center)[
    #text(26pt, weight: "bold")[IMU 里程计教程]
    #v(0.6em)
    #text(14pt, fill: luma(70))[从启动到原理 —— 纯 IMU 如何输出 x / y / yaw]
    #v(1.8em)
    #text(11pt)[
      实机环境：192.168.3.243 · Ubuntu 22.04 · ROS2 Humble \
      IMU 模块：YB IMU（Yahboom 协议，串口 115200） \
      核心节点：`imu_ros2_device/imu_odom`
    ]
    #v(2.6em)
    #text(10pt, fill: luma(100))[
      本教程回答两个问题：#text(weight: "bold")[怎么把它跑起来]，以及 \
      #text(weight: "bold")[它为什么只能这样、不能那样]
    ]
  ]
  #v(1fr)
  #align(center)[
    #text(9pt, fill: luma(120))[生成于 2026-10 · 配套代码见 `~/imu_ws/src/imu_ros2_device/`]
  ]
]

#outline(title: [目录], depth: 2, indent: 1.4em)

#pagebreak()

// ── 1 环境 ───────────────────────────────────────────────────────────
= 硬件与软件环境

== 一句话说明这套系统

IMU 模块通过 USB 串口把姿态数据吐给 `ybimu_driver` 节点，该节点发布成标准 ROS2 话题 `imu/data_raw`；
我们新写的 `imu_odom` 节点订阅这个话题，把原始数据积分成 `nav_msgs/Odometry`，于是你就得到了
#text(weight: "bold")[x、y、yaw 三个量]，外加一条 `odom -> base_link` 的 TF。

#figure(
  kind: table,
  caption: [环境清单],
  table(
    columns: (3.2cm, 1fr),
    align: left + horizon,
    stroke: 0.5pt + luma(190),
    inset: 6pt,
    table.header([*项目*], [*值*]),
    [主机], [`rc@192.168.3.243`，Ubuntu 22.04，Python 3.10.12],
    [ROS2], [Humble，安装于 `/opt/ros/humble`],
    [IMU 串口], [`/dev/myimu` -> 实际 `/dev/ttyUSB0`，115200-8N1，用户已在 `dialout` 组],
    [Python 库], [`YbImuLib`（已 pip 安装）、`pyserial 3.5`],
    [工作空间], [`~/imu_ws`，包名 `imu_ros2_device`],
    [节点], [`ybimu_node`（驱动）+ `imu_odom`（里程计）],
    [话题], [`imu/data_raw` 10 Hz -> `imu/odom` 50 Hz，`/tf`、`/imu/path`],
    [本机开发机], [`192.168.3.247`，与 243 同网段，可直接 DDS 发现话题],
  ),
)

== 数据是怎么一步步产生的

```text
YB IMU 模块 (陀螺+加速度+磁力计+气压计，板载九轴融合)
      │  USB 串口 115200，帧: 7E 23 | LEN | FUNC | DATA | CHECKSUM
      ▼
YbImuSerial 后台线程收帧  →  解析出 accel(g) / gyro(rad/s) / mag(uT) / quat / euler
      ▼
ybimu_driver (10 Hz 定时器)
      ├── imu/data_raw   sensor_msgs/Imu      ← 里程计用的就是它
      ├── imu/mag        sensor_msgs/MagneticField
      ├── euler          std_msgs/Float32MultiArray  [roll, pitch, yaw] 度
      └── baro           std_msgs/Float32MultiArray  [高度, 温度, 气压, 气压差]
      ▼
imu_odom (50 Hz 定时器)
      ├── imu/odom       nav_msgs/Odometry     x, y, yaw + 速度 + 协方差
      ├── imu/path       nav_msgs/Path         轨迹（RViz 看）
      └── /tf            odom → base_link
```

== 串口帧协议（读驱动源码时会遇到）

`YbImuSerialLib.py` 的帧头是 `0x7E 0x23`，第 3 字节是整帧长度，最后一个字节是累加和校验：
`CHECKSUM = (0x7E + 0x23 + LEN + FUNC + sum(DATA)) mod 256`。

#figure(
  kind: table,
  caption: [常用功能码（FUNC）],
  table(
    columns: (2.2cm, 3.4cm, 1fr),
    align: left + horizon,
    stroke: 0.5pt + luma(190),
    inset: 6pt,
    table.header([*FUNC*], [*内容*], [*DATA 布局 / 单位*]),
    [`0x04`], [加速度+角速度+磁力计原始值], [3×int16 accel（g，±16 g）、3×int16 gyro（rad/s，±2000 dps）、3×int16 mag（µT，±800 µT），共 18 B],
    [`0x16`], [四元数], [4×float32 LE，顺序 w x y z，共 16 B],
    [`0x26`], [欧拉角], [3×float32 LE，顺序 roll pitch yaw，#text(fill: rgb("#b3540a"))[弧度]，共 12 B],
    [`0x32`], [气压计], [4×float32：高度(m)、温度、气压、气压差，共 16 B],
    [`0x60`], [设置上报频率], [10~100 Hz],
    [`0x61`], [6/9 轴融合选择], [6 = 只用陀螺+加速度；9 = 加上磁力计],
    [`0x70`/`0x71`], [IMU 校准 / 磁力计校准], [静止校准陀螺加速度；转圈校准磁力计],
  ),
)

#note("单位换算的坑")[
  `get_imu_attitude_data(ToAngle=True)` 返回的是 #text(weight: "bold")[度]，直接传 `False` 才是弧度；
  但 `imu/data_raw` 里的 `orientation` 四元数是没有单位的。另外 `linear_acceleration`
  在本驱动里发的是 #text(weight: "bold")[g，不是 m/s²]，做积分前必须乘 9.80665 —— 我们的
  `imu_odom` 用参数 `accel_in_g` 来处理这件事。
]

#pagebreak()

// ── 2 启动 ───────────────────────────────────────────────────────────
= 快速启动

== 一条命令启动

```bash
ssh rc@192.168.3.243
source /opt/ros/humble/setup.bash
source ~/imu_ws/install/setup.bash
ros2 launch imu_ros2_device imu_odom.launch.py
```

启动成功会看到两行关键日志：

```text
[ybimu_driver-1] [INFO] [...] [ybimu_node]: Open Ybimu Port OK:/dev/myimu
[imu_odom-2]     [INFO] [...] [imu_odom]: IMU 里程计就绪: mode=accel, yaw_source=orientation,
                                订阅 imu/data_raw -> 发布 imu/odom
```

launch 文件默认 `mode:=accel`，也就是 #text(weight: "bold")[x / y / yaw 全输出]。

== 常用变体

```bash
# 只要航向角（x/y 恒为 0，长时间最稳）
ros2 launch imu_ros2_device imu_odom.launch.py mode:=yaw

# 换成别的原始话题名
ros2 launch imu_ros2_device imu_odom.launch.py imu_topic:=imu/raw_data

# 分两个终端手动起（调试方便）
ros2 run imu_ros2_device ybimu_driver
ros2 run imu_ros2_device imu_odom --ros-args -p mode:=accel

# 里程计归零（重新把当前朝向当 0 度）
ros2 service call /reset_odom std_srvs/srv/Trigger
```

== 验证它真的在跑

```bash
ros2 node list                      # 应有 /ybimu_node 和 /imu_odom
ros2 topic list                     # /imu/data_raw /imu/odom /imu/path /tf ...
ros2 topic hz /imu/data_raw         # 期望 ≈ 10.0 Hz
ros2 topic hz /imu/odom             # 期望 ≈ 50.0 Hz
ros2 topic echo /imu/odom --once    # 看 x / y / yaw 有没有数
ros2 run tf2_ros tf2_echo odom base_link
```

实测输出（静止状态，`mode:=accel`）：

```text
x     = 0.0008 m
y     = -0.0011 m
yaw   = 0.01 deg   (qz=0.00013, qw=1.00000)
wz    = 0.0000 rad/s
频率  = 50.001 Hz
```

== 停止

```bash
pkill -f imu_odom
pkill -f ybimu_driver
```

#warn("绝对不要同时起两个驱动")[
  串口是独占资源。我第一次测试时 launch 起了第二个 `ybimu_driver`，而前一个进程还活着，
  日志立刻报：
  `serial.serialutil.SerialException: device reports readiness to read but returned no data
  (device disconnected or multiple access on port?)`
  —— 结果就是数据中断。*停干净旧进程再起新的*，或者永远只用一个 launch 实例。
]

== RViz 可视化

```bash
rviz2
```
依次 `Add`：
- `Odometry` → Topic 选 `/imu/odom`（看箭头朝向和位置）
- `Path` → Topic 选 `/imu/path`（看走过的轨迹）
- `TF` → 看 `odom` 与 `base_link` 两个坐标系

Fixed Frame 填 `odom`。

#pagebreak()

// ── 3 节点实现 ───────────────────────────────────────────────────────
= 节点实现解读

源码位置：`~/imu_ws/src/imu_ros2_device/imu_ros2_device/imu_odom.py`（约 300 行）。

== 整体流程

```text
_on_imu(msg)                     ← 订阅回调，收到一帧原始 IMU 就算一次
  │
  ├─ 第一帧：记录时间基准与起始 yaw（zero_on_start），直接返回
  ├─ dt = 当前时间 - 上一帧时间（dt<=0 或 >1s 直接丢弃，防时间跳变）
  ├─ 航向：yaw = 四元数转出的角 - 起始角        （yaw_source=orientation）
  │        或 yaw += wz * dt                    （yaw_source=gyro）
  └─ 位置：mode=accel 时调用 _integrate_accel()

_publish()                       ← 50 Hz 定时器，把最新状态发出去
  └─ Odometry + TF + Path（每 path_stride 帧记一个轨迹点）
```

== 姿态取自四元数，不是自己解算

```python
def quat_to_yaw(w, x, y, z):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
```

这是标准的 ZYX 欧拉角提取公式。要注意的是，#text(weight: "bold")[我们没有自己做姿态解算]——
YB IMU 模块内部固件已经完成了九轴融合（陀螺 + 加速度 + 磁力计），四元数是它的输出结果，
我们只做数学提取。这也是为什么参数叫 `yaw_source:=orientation` 而不是 "gyro"：
用模块的融合结果，航向有磁力计约束，不会随时间漂。

== 加速度积分的三个关键处理

`_integrate_accel()` 里做了三件事，缺一件都不可用：

*1）扣重力*

```python
scale = 9.80665 if accel_in_g else 1.0
ax, ay, az = ax*scale, ay*scale, az*scale
wx, wy, wz_a = quat_rotate(q, (ax, ay, az))   # 机体 → 世界系
wz_a -= 9.80665                               # 去掉重力分量
```

写成公式就是
$ a_"world" = R(q) dot a_"body" - g $
其中 $R(q)$ 是四元数给出的旋转矩阵，$g = (0, 0, 9.80665) "m/s"^2$。
不扣重力的话，竖直方向会立刻积分出一个 $1/2 g t^2$ 的巨大假位移。

*2）ZUPT（零速修正）*

```python
still = abs(norm(a) - 9.80665) < gtol and all(|gyro| < 0.05)
if still_streak >= 3:
    vx = vy = 0          # 连续 3 帧静止就强制零速
```

车停着的时候"真实速度为 0"是已知的硬约束，直接用它清零速度，能把静止期的漂移彻底消灭。
实测效果：静止 25 秒，位移 0.0008 m 之后被锁死为 0。

*3）速度阻尼*

$ v_(k+1) = v_k e^(-Delta t \/ tau), quad tau = 0.8 "s" $

把每次更新后的速度按指数衰减压一压。代价是位移会被低估，收益是长时间运行不会二次发散。
`velocity_decay_tau` 设为 `<=0` 可关闭。

== 发布的 Odometry 长什么样

```text
header.frame_id  = odom          # 父坐标系
child_frame_id   = base_link     # 子坐标系
pose.position.x / y              # 平面位置（米）
pose.orientation (w,x,y,z)       # 姿态，yaw 藏在里面
pose.covariance[0], [7], [35]    # x, y, yaw 的方差
twist.linear.x / y               # 世界系线速度
twist.angular.z                  # 角速度 wz
```

协方差是给下游（比如 EKF）看的：`mode=yaw` 时 x/y 方差填 `1e6`（等于告诉别人"这两个量未知"），
`mode=accel` 时填 `0.25`；`yaw_source=orientation` 的 yaw 方差 `0.02`，纯陀螺积分则给 `0.5`。

== 参数速查

#figure(
  kind: table,
  caption: [`imu_odom` 全部参数与默认值],
  table(
    columns: (4.1cm, 3.1cm, 1fr),
    align: left + horizon,
    stroke: 0.5pt + luma(190),
    inset: 6pt,
    table.header([*参数*], [*默认*], [*作用*]),
    [`imu_topic`], [`imu/data_raw`], [订阅的原始 IMU 话题],
    [`odom_topic`], [`imu/odom`], [输出的里程计话题],
    [`odom_frame` / `base_frame`], [`odom` / `base_link`], [TF 的父子坐标系],
    [`publish_tf`], [`true`], [是否广播 TF],
    [`publish_rate`], [`50.0`], [发布频率 Hz（积分仍在 IMU 回调里做）],
    [`mode`], [`yaw`（launch 改成 `accel`）], [`yaw` 只给航向；`accel` 额外积分出 x/y],
    [`yaw_source`], [`orientation`], [用四元数（不漂）还是陀螺积分（会漂）],
    [`accel_in_g`], [`true`], [原始加速度是否要乘 9.80665],
    [`zupt_enable`], [`true`], [静止零速修正开关],
    [`zupt_lin_tol`], [`0.06`], [静止判据：|‖a‖−1g| 阈值（g）],
    [`zupt_gyro_tol`], [`0.05`], [静止判据：陀螺阈值（rad/s）],
    [`velocity_decay_tau`], [`0.8`], [速度阻尼时间常数（s），`<=0` 关闭],
    [`zero_on_start`], [`true`], [启动时把当前朝向记为 0 度],
    [`path_stride`], [`5`], [每 N 次发布记一个轨迹点],
  ),
)

#pagebreak()

// ── 4 原理 ───────────────────────────────────────────────────────────
= 原理：IMU 到底能定位到什么程度

== IMU 给你的三种原始信息

+ *角速度* $omega$（rad/s）：三轴陀螺，积分得角度。优点是短期极准、不受外界干扰；
  缺点是零偏累积，纯积分会漂。
+ *线加速度* $a$（这里以 g 给出，用时乘 9.80665）：三轴加速度计，静止时读到的是
  #text(weight: "bold")[支撑力而不是运动加速度]（即 1g 重力），所以必须先旋转到世界系再减掉 $g$。
+ *磁场* $B$（µT）：提供绝对航向参考（类似指南针），让 yaw 不随时间漂；
  缺点是附近有电机、铁磁物时会被干扰。

九轴融合 = 用加速度计校正陀螺的俯仰/横滚漂移，用磁力计校正航向漂移。
YB 模块把这些都做在固件里了，`set_algo_type(9)` 选择九轴，`set_algo_type(6)` 退化为六轴
（此时 yaw 只能靠陀螺积分，会慢漂）。

== 为什么纯 IMU 定不了位置

惯性定位的数学本质是两次积分：

$ v_k = v_0 + sum_i a_i Delta t, quad x_k = x_0 + sum_j v_j Delta t $

任何误差都要被积分两次。陀螺零偏、加速度计零偏、姿态角误差、噪声，都会随着
#text(weight: "bold")[$t^2$] 增长。举个数量级的例子：加速度计有 0.01 m/s² 的残余零偏，
10 秒后位置误差约 $1/2 times 0.01 times 100 = 0.5$ m，60 秒后就是约 18 m —— 而你的场地可能只有几米见方。

#warn("这是物理限制，不是代码没写好")[
  任何"纯 IMU 定位"方案，无论是手写积分、互补滤波还是卡尔曼滤波，只要没有外部参考
  （轮速、视觉、磁力计航向、UWB、GNSS），长时间都必然发散。本节点的 ZUPT 与速度阻尼
  是工程上"把发散摁住"的手段，不是消除根本原因。
]

那么实际工程怎么解？答案是 *航迹推算 + IMU* 分工：

#figure(
  kind: table,
  caption: [IMU 与里程计的合理分工],
  table(
    columns: (3cm, 3.4cm, 1fr),
    align: left + horizon,
    stroke: 0.5pt + luma(190),
    inset: 6pt,
    table.header([*物理量*], [*谁来提供*], [*原因*]),
    [位置 x, y], [轮速里程计（航迹推算）], [轮子转过的弧长是绝对位移参考，不随时间漂],
    [航向 yaw], [IMU 九轴融合], [磁力计给绝对航向；轮速差分的 yaw 在打滑时会错],
    [线速度 / 角速度], [两者皆可，IMU 的 $omega_z$ 更实时], [IMU 无延迟，轮速有编码器分辨率量化],
    [高度 z], [气压计输出的第 `0` 项], [平面场地可忽略],
  ),
)

ROS2 里的标准做法是 `robot_localization` 包的 EKF，把 `/odom_raw`（轮速）和 `/imu/data`
（IMU）融合成 `/odom`。项目里现成的配置文件
`ROS应用/ROS2/yahboomcar_ws/src/robot_localization/params/ekf_A1.yaml` 中，
`imu0_config` 恰好只打开 yaw 与 yaw 角速度两项 —— 这就是"IMU 负责朝向、里程计负责位移"的配置化表达。

== 两种模式的取舍

#figure(
  kind: table,
  caption: [`mode:=yaw` 与 `mode:=accel` 对比],
  table(
    columns: (3.2cm, 1fr, 1fr),
    align: left + horizon,
    stroke: 0.5pt + luma(190),
    inset: 6pt,
    table.header([*维度*], [*`yaw` 模式*], [*`accel` 模式*]),
    [输出], [yaw 恒有，x/y 恒为 0], [x、y、yaw 全有],
    [长时间稳定性], [#text(fill: rgb("#1e7a37"))[好]（磁力计约束）], [#text(fill: rgb("#b3540a"))[差]（二次积分必漂）],
    [适用场景], [判断朝向、原地转向、航向闭环], [短时位移估计、演示、静止不漂],
    [协方差], [x/y 填 1e6 表示"未知"], [x/y 填 0.25],
    [风险], [强磁干扰时 yaw 会偏], [运动中几十秒后位置会偏],
  ),
)

== 坐标系与 TF

- `odom`：里程计世界系，原点是节点启动那一刻的位置与朝向（`zero_on_start:=true`）。
- `base_link`：车体中心。`imu_odom` 以 50 Hz 广播 `odom -> base_link` 变换，
  任何 TF 消费者（Nav2、RViz、自己写的节点）都能实时拿到。
- 变换内容就是平移 $(x, y, 0)$ 加上绕 Z 旋转 $psi$：

$ T = mat(delim: "|", cos psi, -sin psi, x; sin psi, cos psi, y; 0, 0, 1) $

#pagebreak()

// ── 5 实测与坑 ───────────────────────────────────────────────────────
= 实测记录与踩过的坑

== 实测数据

#figure(
  kind: table,
  caption: [在 192.168.3.243 上的实测结果],
  table(
    columns: (4.6cm, 3.4cm, 1fr),
    align: left + horizon,
    stroke: 0.5pt + luma(190),
    inset: 6pt,
    table.header([*测试项*], [*结果*], [*说明*]),
    [`/imu/data_raw` 频率], [`average rate: 10.000`], [驱动定时器 0.1s，抖动 std dev ≈ 0.00013s],
    [`/imu/odom` 频率], [`average rate: 50.001`], [launch 里 `publish_rate: 50.0`],
    [静止 25s 位移（accel 模式）], [`x=0.0005 m`, `y=-0.0009 m`], [ZUPT 生效，随后被锁死为 0],
    [一帧 odom 输出], [`x=0.0008`, `y=-0.0011`, `yaw=0.01°`], [静止、`zero_on_start` 生效],
    [`/euler` 原始欧拉角], [`roll=-79.6° pitch=-11.8° yaw=70.25°`], [#text(weight: "bold")[模块是斜装的，见下]],
    [TF], [`odom -> base_link` 50 Hz], [`tf2_echo` 可持续读到],
  ),
)

== 坑一：模块斜装，yaw 不等于车体航向

静止时模块读出 `roll = -79.6°, pitch = -11.8°`，说明它不是平放的，而是几乎侧立着装的。
这时四元数解出的 yaw 是绕 #text(weight: "bold")[模块自身 Z 轴]的角，与车体的偏航角
之间隔着一个大角度的安装旋转 —— 车转 90°，这里读到的不会是 90°，三个轴的分量是混着的。

两种解决办法：

+ #text(weight: "bold")[硬件解决]：把模块平放，让静止时 roll/pitch 接近 0。最省事，优先考虑。
+ #text(weight: "bold")[软件解决]：标定安装外参。先在静止状态下求出模块坐标系到车体坐标系的
  固定旋转 $q_"mount"$，然后在取 yaw 之前先做一次合成：
  $ q_"corrected" = q_"mount" times q_"imu" $
  再从 $q_"corrected"$ 里提取 yaw。这一步尚未实现在 `imu_odom` 里，是后续待办。

== 坑二：不要起两个驱动抢串口

见第 2 章的警告框。症状是 `SerialException ... multiple access on port`，数据中断。

== 坑三：pkill 会杀掉自己的命令行

调试时写 `pkill -f imu_odom`，如果这条命令本身的命令行里就含有 `imu_odom` 字样
（比如 `ros2 run imu_ros2_device imu_odom ...`），pkill 会把执行它的 shell 一起杀掉，
表现为命令莫名返回、什么输出都没有。解决办法是把操作写进 `.sh` 脚本再执行，
让命令行只出现 `bash /tmp/test.sh`。

== 坑四：ROS2 环境变量必须先 source

```bash
source /opt/ros/humble/setup.bash      # 不执行则 ros2 命令不存在
source ~/imu_ws/install/setup.bash     # 不执行则找不到 imu_ros2_device 包
```
在脚本里还要注意 `set -u` 会与 ROS 的 setup.bash 冲突（报
`AMENT_TRACE_SETUP_FILES: 未绑定的变量`），要么去掉 `set -u`，要么先 `set +u` 再 source。

== 故障排查

#figure(
  kind: table,
  caption: [常见故障与定位方法],
  table(
    columns: (5.2cm, 1fr),
    align: left + horizon,
    stroke: 0.5pt + luma(190),
    inset: 6pt,
    table.header([*现象*], [*排查*]),
    [`Open Ybimu Port FAIL`], [串口是否被占用（`fuser /dev/ttyUSB0`）、是否在 `dialout` 组、设备节点是否存在],
    [`/imu/odom` 有话题但频率 0], [`ros2 topic hz /imu/data_raw` 是否也是 0 —— 上游驱动没出数则先查驱动],
    [yaw 不动], [模块是否静止？`ros2 topic echo /euler` 对照看原始欧拉角有没有变化],
    [x/y 越跑越大], [正常现象（纯 IMU 积分）；确认 ZUPT 开着、`accel_in_g:=true`],
    [跨机看不到话题], [两台机器 `echo $ROS_DOMAIN_ID` 必须一致、同一网段、防火墙放行 UDP 多播],
    [编译后改动不生效], [`cd ~/imu_ws && colcon build --packages-select imu_ros2_device` 后需重新 source install],
  ),
)

#pagebreak()

// ── 6 进阶 ───────────────────────────────────────────────────────────
= 进阶路线

== 下一步建议（按性价比排序）

+ *标定安装外参*：解决斜装问题，让 yaw 真正等于车体航向。收益最大，工作量半天。
+ *加轮速里程计*：用 `ROS应用/ROS2/yahboomcar_ws/src/yahboomcar_base_node/src/base_node_A1.cpp`
  的航迹推算公式（$Delta x = v cos psi Delta t$，$Delta y = v sin psi Delta t$）发 `/odom_raw`。
+ *EKF 融合*：装 `ros-humble-robot-localization`，照抄 `ekf_A1.yaml`，把 `odom0=/odom_raw`
  和 `imu0=/imu/data` 融成最终 `/odom`。这是"能用的定位"的标准答案。
+ *接入检测程序*：`upper_machine.py` 里开一个 rclpy 后台线程订 `/imu/odom`，
  拿 `pose.position.x/y` 与四元数转出的 yaw，用于给视觉结果定向。
+ *绝对位置校正*：场地上的二维码/标志物给出全局参考，看到时调用
  `/reset_odom` 或直接发 `ResetLocalization(x, y, yaw)`，消除累积误差。

== 相关文件索引

#figure(
  kind: table,
  caption: [代码与配置位置],
  table(
    columns: (6.6cm, 1fr),
    align: left + horizon,
    stroke: 0.5pt + luma(190),
    inset: 6pt,
    table.header([*路径*], [*作用*]),
    [`~/imu_ws/src/imu_ros2_device/imu_ros2_device/imu_odom.py`], [本教程主角：IMU 里程计节点],
    [`~/imu_ws/src/imu_ros2_device/imu_ros2_device/ybimu_driver.py`], [串口驱动，发布 `imu/data_raw`],
    [`~/imu_ws/src/imu_ros2_device/launch/imu_odom.launch.py`], [一键启动（驱动 + 里程计）],
    [`~/imu/YbImuLib/YbImuLib/YbImuSerialLib.py`], [串口收帧、单位换算、校准命令],
    [`~/imu_ws/src/.../config/imu_filter_param.yaml`], [Madgwick 姿态滤波参数（备用方案）],
    [`项目仓库 ROS应用/ROS2/.../params/ekf_A1.yaml`], [轮速+IMU 的 EKF 融合配方（进阶）],
    [`项目仓库 多主控通讯/STM32/USART/BSP/mcu_model/imu_uart_driver.c`], [下位机侧的同款驱动（MCU 用）],
    [`项目仓库 imu校准/IMU_calibration/`], [陀螺仪/磁力计/温度校准脚本],
  ),
)

== 常用命令速查

```bash
# 启动 / 停止
ros2 launch imu_ros2_device imu_odom.launch.py [mode:=yaw|accel] [imu_topic:=...]
pkill -f imu_odom; pkill -f ybimu_driver

# 观测
ros2 topic hz /imu/odom
ros2 topic echo /imu/odom --once
ros2 topic echo /euler --once            # 模块原始 roll/pitch/yaw（度）
ros2 run tf2_ros tf2_echo odom base_link
ros2 service call /reset_odom std_srvs/srv/Trigger

# 重建
cd ~/imu_ws && colcon build --packages-select imu_ros2_device && source install/setup.bash

# 改上报频率 / 校准（在 Python 里）
#   imu.set_report_rate(100)      10~100 Hz
#   imu.set_algo_type(9)          九轴融合，yaw 有磁力计约束
#   imu.calibration_imu()         静止校准陀螺+加速度
#   imu.calibration_mag()         转圈校准磁力计
```

#tip("一句话总结")[
  启动只需 `ros2 launch imu_ros2_device imu_odom.launch.py`；
  #text(weight: "bold")[yaw 可以放心长期用]（九轴融合 + 磁力计），
  #text(weight: "bold")[x/y 只能短时用]（加速度二次积分 $t^2$ 发散，ZUPT 和阻尼只是权宜之计）；
  要真正的位置，必须把轮速里程计用 EKF 和 IMU 融起来 —— 而第一步，先把那个
  $-79.6°$ 的安装角解决掉。
]
