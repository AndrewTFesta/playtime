"""
@title

@description

"""
import argparse
from pathlib import Path

from transformers import AutoTokenizer, AutoFeatureExtractor, Wav2Vec2Processor
from transformers import pipeline

from playtime import project_properties

MODEL_TASK = 'automatic-speech-recognition'
# https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=trending
MODEL_NAMES = [
    'facebook/wav2vec2-base-960h',
    'openai/whisper-large-v2'
]
AUDIO_EXTS = [
    'mp3',
    'mp4'
]
AUDIO = [f"https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/{i}.flac" for i in range(1, 5)]
MLK_AUDIO = str(Path(project_properties.data_dir, 'mlk.flac'))
DANI_AUDIO = [
    str(each_file)
    for each_ext in AUDIO_EXTS
    for each_file in Path(project_properties.data_dir, 'The Fantasy Realm').glob(f'**/*.{each_ext}')
]


def fine_tune():
    """

    :return:
    """
    model_checkpoint = "facebook/wav2vec2-large-xlsr-53"
    # after defining a vocab.json file you can instantiate a tokenizer object:
    tokenizer = AutoTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
    processor = Wav2Vec2Processor.from_pretrained(feature_extractor=feature_extractor, tokenizer=tokenizer)
    return


def dani_data():
    for each_audio in DANI_AUDIO:
        yield each_audio


def main(main_args):
    transcriber = pipeline(model=MODEL_NAMES[0], task=MODEL_TASK, device_map='auto', batch_size=2, return_timestamps=True, chunk_length_s=30)
    print(transcriber.model.name_or_path)

    # UserWarning: 1
    # Torch was not compiled with flash attention
    # Due to a bug fix in https://github.com/huggingface/transformers/pull/28687 transcription using a multilingual Whisper will default to language detection followed
    # by transcription instead of translation to English.This might be a breaking change for your use case.
    # If you want to instead always translate your audio to English, make sure to pass `language='en'`.

    # audio files must be passed as str, not Path
    for each_transcription in transcriber(dani_data()):
        print(f'{each_transcription}')

    # transcribed = transcriber(DANI_AUDIO)
    # for each_transcription in transcribed:
    #     print(f'{each_transcription}')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
