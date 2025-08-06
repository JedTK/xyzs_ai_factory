from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

model_dir = "iic/SenseVoiceSmall"

model = AutoModel(
    model=model_dir,
    trust_remote_code=True,
    vad_kwargs={"max_single_segment_time": 30000},
    device="cpu",
    hub="hf",
    disable_update=True,
)

# en
res = model.generate(
    input="/Users/jedwong/Downloads/16k16bit.wav",
    cache={},
    language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
    use_itn=True,
    batch_size_s=60,
    merge_vad=True,  #
    merge_length_s=15,
)
text = rich_transcription_postprocess(res[0]["text"])
print(text)

