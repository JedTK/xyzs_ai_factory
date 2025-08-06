from xyzs_ai_factory.Module.ASR import XFunASR
from xyzs_ai_factory.Module.BaseFactory.ASR import ASRFactory

asr_factory = ASRFactory()
asr_factory.register(module_name="XFunASR", cls=XFunASR)

# 兼容两种调用方式一：
# asr_factory.init_instance(module_name="XFunASR",
#                           model="/Users/jedwong/Work/JProject/AI/model/FunAudioLLM/SenseVoiceSmall",  # 可选，默认就是这个
#                           trust_remote_code=True,  # 可选，是否信任远程代码
#                           vad_kwargs={"max_single_segment_time": 30000},  # 可选，VAD 参数
#                           device="cpu",  # 或 "cuda"（如果有 GPU）
#                           hub="hf",
#                           )
#
# result = asr_factory.get_instance("XFunASR").transcribe(audio_file="/Users/jedwong/Downloads/16k16bit.wav",
#                                                         lang="zh",
#                                                         cache={},
#                                                         language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
#                                                         use_itn=True,
#                                                         batch_size_s=60,
#                                                         merge_vad=True,  #
#                                                         merge_length_s=15, )

# 兼容两种调用方式二：
asr_factory.init_instance(module_name="XFunASR", config={
    "model": "/Users/jedwong/Work/JProject/AI/model/FunAudioLLM/SenseVoiceSmall",  # 可选，默认就是这个
    "trust_remote_code": True,  # 可选，是否信任远程代码
    "vad_kwargs": {"max_single_segment_time": 30000},  # 可选，VAD 参数
    "device": "cpu",  # 或 "cuda"（如果有 GPU）
    "hub": "hf",
})

text = asr_factory.get_instance("XFunASR").transcribe(audio_file="/Users/jedwong/Downloads/16k16bit.wav", config={
    "lang": "zh",
    "cache": {},
    "language": "auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
    "use_itn": True,
    "batch_size_s": 60,
    "merge_vad": True,  #
    "merge_length_s": 15,
})
print("识别文本：", text)

text1 = asr_factory.get_instance("XFunASR").transcribe(audio_file="/Users/jedwong/Downloads/16k16bit.wav", config={
    "lang": "zh",
    "cache": {},
    "language": "auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
    "use_itn": True,
    "batch_size_s": 60,
    "merge_vad": True,  #
    "merge_length_s": 15,
})

print("识别文本：", text1)
