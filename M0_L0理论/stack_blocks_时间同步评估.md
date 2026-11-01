# Ego 采集数据时间同步评估：帧级对齐「修复 vs 重采」

- 评估日期：2026-10-22
- 数据位置：`/mnt/mnt/data/dataset/ego/test/`（4 个 take）
- 背景：task.json 中 `timing_quality: host_events_not_frame_sync`，即事件时间戳未与视频帧级对齐。评估目标：能否从现有文件重建帧级对齐，还是必须改采集器重采。

**结论（先行）：帧级对齐不可修复。** 现有文件缺失「事件 unix 时刻 ↔ 视频帧」信息链中的全部 wall-clock 锚点：ffmpeg 未记录启动 wall time、mp4 无 creation_time、PTS 被归零且为媒体服务器（ZLMediaKit）重基准、RTSP 链路延迟完全未记录。唯一能推算的近似锚点存在 ±0.2~0.8s 的不可消除误差（≈10~36 帧 @45fps），不满足帧级配对要求。**推荐改采集器重采**，采集器改动量小（见 §6）。

---

## 1. task.json 总览（4 个 take）

| take | task_slug | status | started_at_unix | finished_at_unix | error |
|---|---|---|---|---|---|
| 20261017_121956_645999 | stack_blocks | **failed** | 1792210796.646 | 1792210797.039 | 浏览器播放失败：参考视频超过 5 秒未播放 |
| 20261017_122001_322888 | stack_blocks | complete | 1792210801.323 | 1792210820.982 | — |
| 20261017_132427_036272 | pick_up_paper_rolls | **failed** | 1792214667.036 | 1792214667.579 | 浏览器播放失败：参考视频超过 5 秒未播放 |
| 20261017_132434_574206 | pick_up_paper_rolls | complete | 1792214674.574 | 1792214695.655 | — |

要点：
- 4 个 take 中 2 个为失败 take（启动后 <1s 即因浏览器参考视频播放失败而终止），其 mp4 仅 1.18s / 0.69s，无研究价值。有配对价值的只有 2 个 complete take。
- 所有 take 均 `timing_quality: host_events_not_frame_sync`。
- `camera: "left"`，但双目 left.mp4/right.mp4 均被录制（两路流都录了）。

## 2. events.jsonl 解析

### 2.1 事件类型与时序（以 complete take 20261017_122001_322888 为例）

```
take_created
pedal                  +0.006s  {action: x, state: ready}
recorder_started       +0.000s  {stream: right, command: [ffmpeg, ..., rtsp://192.168.31.238:8554/ego_cam/right, ..., -c, copy, ...]}
recorder_started       +0.000s  {stream: left}
first_media_progress   +0.123s  {stream: right, media_time_us: 11}
first_media_progress   +0.000s  {stream: left,  media_time_us: 11}
reference_play_requested +0.000s
reference_play_ack     +0.150s  {position_sec: 0.0}
pedal                  +18.833s {action: x, state: recording}
reference_paused       +0.100s  {position_sec: 18.2}
recorders_stopped      +0.064s  {returncodes: {right: 0, left: 0}}
take_finished          +0.383s  {status: complete}
```

事件类型全集：`take_created, pedal(ready/recording), recorder_started(×2), first_media_progress(×2), reference_play_requested, reference_play_ack, reference_eof, reference_paused, recorders_stopped, take_finished`。失败 take 缺少 `reference_play_ack / reference_eof / reference_paused / pedal(recording)`，事件流结构与其 failed 状态一致——完整性本身无异常。

### 2.2 时钟质量

- 每个事件带 `monotonic_ns` 与 `unix_ns` 双时间戳。两时钟均严格单调递增。
- 单 take 内 `unix_ns - monotonic_ns` 偏移量散布仅 682~3427 ns（python 计算），说明双时钟同源、转换一致，**事件流自身的时钟质量是好的**。
- 问题不在事件流内部，而在事件流与视频帧之间没有任何共同锚点（见 §4）。

## 3. 视频与 ffmpeg 日志检查

### 3.1 ffprobe 结果（每路流）

```
codec_name=hevc, 1920x1920, r_frame_rate=45/1, time_base=1/90000
start_pts=0, start_time=0.000000
format_tags: title=Streamed by ZLMediaKit(...), encoder=Lavf58.76.100
             ← 无 creation_time 标签
```

| take | 流 | 包数 | duration | avg_frame_rate |
|---|---|---|---|---|
| 121956 (failed) | left/right | 54 | 1.177s | 97200/2119 ≈ 45.88 |
| 122001 (complete) | left/right | 890 | 19.755s | 8010000/177797 ≈ 45.05 |
| 132427 (failed) | left/right | 32 | 0.688s | 144000/3097 ≈ 46.50 |
| 132434 (complete) | left/right | 966 | 21.444s | 4347000/96499 ≈ 45.05 |

- 所有文件 PTS 从 0 开始（ffmpeg 未加 `-copyts`，默认将时间戳归零）。
- **无 creation_time / 任何 wall-clock 元数据标签**。
- 双目两路：包数、duration、首包 PTS 序列完全一致（0, 0.000011, 0.022000, …），双目间相对同步良好。
- 帧间隔：45fps 下 delta 为 0.022s/0.023s 交替（90MHz 时基取整），无丢帧迹象；但 4 个 take 首包 PTS 模式完全相同（0, 0.000011, 0.022…），说明 PTS 是 ZLMediaKit 媒体服务器按接收/转发顺序重基准生成的，**不携带相机采集时刻信息**。

### 3.2 ffmpeg 录制命令（来自 events.jsonl 的 recorder_started）

```
ffmpeg -hide_banner -loglevel warning -rtsp_transport tcp \
  -i rtsp://192.168.31.238:8554/ego_cam/{left,right} -map 0:v:0 -c copy -n \
  -movflags +frag_keyframe+empty_moov+default_base_moof \
  -stats_period 0.2 -progress pipe:1 <out>.mp4
```

关键缺失：**没有 `-use_wallclock_as_timestamps`、没有 `-copyts`、没有 `-metadata creation_time=...`**。

### 3.3 ffmpeg 日志内容

每个 `*.ffmpeg.log` 只有 2 行警告，无任何时间信息：

```
[mp4 @ ...] Timestamps are unset in a packet for stream 0. ...
[mp4 @ ...] Non-monotonous DTS in output stream 0:0; previous: 0, current: 0; changing to 1. ...
```

`-loglevel warning` 下不打印启动横幅与 wall-clock；日志中**不存在任何 wall-clock 锚点**。

## 4. 核心判断：事件 unix_ns ↔ 视频帧的映射是否可建立？

需要的信息链：`事件 unix 时刻 ↔ 视频中该时刻对应的帧`，即需要知道「帧 i 的 PTS ↔ 某个 host wall-clock 时刻」的至少一个锚点，外加帧的相机采集时刻与 host 接收时刻之间的延迟。

逐项检查：

| 环节 | 状态 | 证据 |
|---|---|---|
| ffmpeg 日志记录 start wall time | ❌ 缺失 | 日志仅 2 行警告（§3.3） |
| mp4 creation_time / 元数据锚点 | ❌ 缺失 | format_tags 无 creation_time（§3.1） |
| PTS 携带 wall-clock（-copyts / -use_wallclock_as_timestamps） | ❌ 未启用 | 命令行无此参数；所有文件 start_time=0（§3.1、§3.2） |
| RTSP 流延迟（相机采集→host 接收） | ❌ 未记录 | 无任何 RTP/RTCP 时间戳或延迟日志；PTS 由 ZLMediaKit 重基准 |
| 每帧 PTS 导出 | ❌ 未做 | 只能事后 ffprobe 重建，但只有相对 PTS |

**唯一可用的近似锚点**：`first_media_progress` 事件（host unix 时刻，ffmpeg 首次通过 `-progress` 上报输出，`media_time_us=11` ≈ 已写出 2 个包）。理论上可近似认为「帧 0 ≈ first_media_progress 时刻前若干毫秒」。但该锚点有三级不可消除的误差：

1. **启动窗口不确定**：`recorder_started → first_media_progress` 为 123~323ms（含 ffmpeg 进程启动、RTSP TCP 握手、服务器缓冲），帧 0 落在此窗口内的具体位置未知。
2. **采集→接收延迟未知**：RTSP 经 ZLMediaKit 中转，相机采集时刻到 host 写盘之间通常有 100~500ms+ 延迟，且逐帧抖动，完全未记录。
3. **缓冲证据**：mp4 duration 比 host 侧 `recorder_started → recorders_stopped` 窗口**长出 0.15~0.79s**（逐 take 不同：+0.790 / +0.485 / +0.152 / +0.705s），说明管道内有可变的缓冲帧被冲刷进文件——每个 take 的零点偏移都不同，无法用固定常数修正。

```
take           recorder span   mp4 duration   diff
121956         0.387s          1.177s         +0.790s
122001         19.270s         19.755s        +0.485s
132427         0.537s          0.688s         +0.152s
132434         20.739s         21.444s        +0.705s
```

结论：即使接受近似锚点，误差也在 ±0.2~0.8s 量级，即 45fps 下 ±10~36 帧。与 teleop 机器人数据做帧级配对（每帧 22ms）时，该误差不可接受，且无法验证修正是否正确。**帧级映射不可建立**。

（注：双目 left/right 之间相对同步可由文件内容保证——两路包数与 PTS 序列一致；缺的是与 host 事件钟的绝对对齐。）

## 5. 外部路径探测

task.json 中 `teleop_dataset`、`reference_video`、`backup.path` 均指向 `/media/dex/Expansion/...`（采集机本地挂载）：

```
$ ls /media/dex/Expansion
ls: cannot access '/media/dex/Expansion': No such file or directory
```

本机 `/media` 下无 `dex` 目录，所有 teleop 配对数据与备份路径在本机均不存在。这也意味着无法借助 teleop 侧的帧时间戳做跨域反推对齐；且失败 take 的 backup 无法本机核验。

## 6. 结论与建议

### 6.1 可修复性判定：**不可修复（帧级）**

- 缺的不是单一环节，而是信息链上的全部 wall-clock 锚点（ffmpeg 启动 wall time、creation_time、wall-clock PTS、RTSP 延迟、每帧 wall time 导出），事后无法从文件内容重新发明这些信息。
- 仅有 2 个 complete take 有配对价值，重采成本低；投入修复只能得到 ±10~36 帧的不确定对齐，对配对研究无价值。

### 6.2 采集器修改建议（改完即可重采）

按优先级，在采集侧增加：

1. **ffmpeg 加 `-use_wallclock_as_timestamps 1`**（RTSP 输入端）：每个包的 PTS 直接变为 host 接收的 unix 时间，事后 `ffprobe -show_packets` 即可得到逐帧 wall-clock。这是最核心的一行改动。（可叠加 `-copyts` 保留原始时间轴。）
2. **录制起始 wall-clock 锚点**：在 `recorder_started` 事件中已记录了命令，再补记 `start_unix_ns`（进程 spawn 时刻已有，建议显式写入 mp4：`-metadata creation_time=<ISO8601>`）。
3. **RTSP 延迟记录**：周期性解析 RTCP Sender Report（RTP 时间戳 ↔ NTP 时刻映射）并写入 events.jsonl，或在 ZLMediaKit 侧开启时间戳透传；最低限度在采集启动时测一次流延迟并记录。
4. **每帧 PTS 导出**：录制结束后自动运行 `ffprobe -show_entries packet=pts_time` 存为 `frames.csv` 随 take 归档，避免下游重复解析。
5. **日志级别**：`-loglevel warning` 改为 `-loglevel info`（或单独 stderr 全量落盘），保留启动横幅（含输入流时间基信息）。
6. events.jsonl 的 monotonic_ns + unix_ns 双时钟设计是好的，保留。

### 6.3 重采建议

- 直接重采 2 个任务（stack_blocks ep0、pick_up_paper_rolls ep1）各 1 个 take 即可覆盖本次损失；现有 4 个 take 中 2 个 failed、2 个 complete 但对齐不可修复，均无帧级配对价值，可仅作采集流程回归测试样本保留。
- 修复采集器后先采 1 个验证 take，用「已知 wall-clock 事件（如 pedal）↔ 帧 PTS」核对端到端误差 < 1 帧，再正式采集。

## 附：关键命令

```bash
# task.json
python3 -m json.tool <take>/task.json
# 事件时钟一致性 / 间隔
python3  # 见 §2 脚本：unix_ns-monotonic_ns 散布 682~3427ns，双时钟严格单调
# 视频结构
ffprobe -v error -show_entries format=duration,start_time:stream=codec_name,avg_frame_rate,start_pts,start_time <take>/left.mp4
ffprobe -v error -count_packets -show_entries stream=nb_read_packets <take>/left.mp4
# 首包 PTS（4 个 take 均为 0, 0.000011, 0.022000 … → ZLMediaKit 重基准）
ffprobe -v error -select_streams v:0 -show_entries packet=pts_time -of csv=p=0 <take>/left.mp4 | head -3
# 外部路径
ls /media/dex/Expansion   # No such file or directory
```
