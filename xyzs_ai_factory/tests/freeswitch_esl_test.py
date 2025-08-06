import gevent
import redfs
import traceback
import threading
import os


"""
                 ┌──────────────┐
                 │  用户拨打电话  │
                 └─────┬────────┘
                       ↓
              ┌───────────────┐
              │ FreeSWITCH    │
              │  mod_audio_fork│
              └────┬────────────┘
                   ↓ WebSocket 音频推流
        ┌────────────────────────────┐
        │  Whisper/Vosk/ASR 服务     │
        │  实时识别语音→转文字       │
        └──────┬────────────┬────────┘
               │            │
               ↓            ↓
         AI模型生成回答   判断是否转人工
               ↓
      TTS 转音频 → 播放语音（uuid_broadcast）
"""


# ESL 配置
ESL_HOST = 'sip.genkigo.net'
ESL_PORT = 8021
ESL_PASSWORD = 'dc956n3gFvikfjNUlrpGDhaHyLRWeXmS'
ESL_TIMEOUT = 10

WELCOME_SOUND_PATH = 'say:您好，欢迎致电智能客服，请说出您的问题'


def asr_process(uuid, filepath):
    """
    模拟语音识别过程，替换为真实的 ASR 调用
    """
    print(f"[ASR] 开始识别通话 {uuid} 的录音文件 {filepath}")
    # TODO：调用本地/云端语音识别接口
    try:
        # 模拟识别结果
        text = f"【模拟转写】用户 {uuid} 的语音内容：你好，我想查一下余额"
        print(text)
    except Exception:
        print("ASR 错误:", traceback.format_exc())

"""
| 英文事件名                     | 中文说明                        |
| ------------------------- | --------------------------- |
| `CHANNEL_CREATE`          | 通道创建（即将开始呼叫）                |
| `CHANNEL_DESTROY`         | 通道销毁（通话彻底结束）                |
| `CHANNEL_ANSWER`          | 被叫接听通话                      |
| `CHANNEL_HANGUP`          | 挂断通话（任一方挂断）                 |
| `CHANNEL_HANGUP_COMPLETE` | 挂断流程完全完成（资源释放）              |
| `CHANNEL_BRIDGE`          | 呼叫双方已桥接成功（开始对话）             |
| `CHANNEL_UNBRIDGE`        | 呼叫桥接断开                      |
| `DTMF`                    | 收到用户按键（DTMF信号）              |
| `RECORD_START`            | 开始录音                        |
| `RECORD_STOP`             | 停止录音                        |
| `PLAYBACK_START`          | 开始播放音频（如 TTS、提示音）           |
| `PLAYBACK_STOP`           | 播放音频结束                      |
| `CUSTOM`                  | 自定义模块事件（如 mod\_audio\_fork） |
| `API`                     | 调用 `api` 命令时的返回             |
| `LOG`                     | 日志级别事件                      |
| `ALL`                     | 所有事件（监听时使用）                 |
"""
def on_event(event):
    try:
        headers = event.headers

        event_name = headers.get('Event-Name')
        uuid = headers.get('Unique-ID')
        # 通话通道名称
        channel_name = headers.get('Channel-Name')
        # print(f"[事件] {name}, UUID={uuid}, Direction={call_dir}")

        # 接通电话后
        if event_name == 'CHANNEL_ANSWER':
            # 主叫号码 (发起呼叫的号码)
            calling_party_number = headers.get('Caller-Caller-ID-Number')

            # 被叫号码 (呼叫的目标号码)
            called_party_number = headers.get('Caller-Destination-Number')

            print(f"[处理] 通话接通，UUID={uuid}, 主叫={calling_party_number}, 被叫={called_party_number}")

            # 播放欢迎语（可用 say: 或 playback）
            conn.send(f'api uuid_broadcast {uuid} "{WELCOME_SOUND_PATH}" both')

            # 启动录音，路径可设为指定目录
            record_path = f"/tmp/{uuid}.wav"
            conn.send(f'api uuid_record {uuid} start {record_path}')
            print(f"[录音] 已开始录音：{record_path}")

        elif event_name == 'CHANNEL_HANGUP':
            print(f"[处理] 通话挂断，UUID={uuid}")
            record_path = f"/tmp/{uuid}.wav"
            if os.path.exists(record_path):
                threading.Thread(target=asr_process, args=(uuid, record_path)).start()

    except Exception:
        print("处理事件异常:", traceback.format_exc())


def main():
    global conn
    conn = redfs.InboundESL(
        host=ESL_HOST,
        port=ESL_PORT,
        password=ESL_PASSWORD,
        timeout=ESL_TIMEOUT
    )

    try:
        conn.connect()
        conn.register_handle('*', on_event)
        conn.send('EVENTS PLAIN ALL')
        print("✅ 已连接 FreeSWITCH，并开始监听所有事件")

        while True:
            gevent.sleep(1)
    except KeyboardInterrupt:
        print("用户中断，退出程序")
        conn.stop()
    except Exception as ex:
        print("连接失败:", ex, traceback.format_exc())


if __name__ == '__main__':
    main()
