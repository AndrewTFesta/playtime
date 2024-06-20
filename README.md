Playtime
======

Notes and important documentation I've picked up while working with various libraries on various projects, or even just playing around.

# Virtual Environment

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

# Documents

`pip install python-docx`

# Huggingface

## Datasets

[Setup and installation](https://huggingface.co/docs/datasets/en/installation)

I primarily use pytorch, but you can use either pytorch, tensorflow, or jax.

```
pip install datasets
python -c "from datasets import load_dataset; print(load_dataset('squad', split='train')[0])"
```
```
{
  'answers': {
    'answer_start': [515],
    'text': ['Saint Bernadette Soubirous']
  },
  'context': 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.',
  'id': '5733be284776f41900661182',
  'question': 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?',
  'title': 'University_of_Notre_Dame'
}
```

## Audio

- [Load audio data](https://huggingface.co/docs/datasets/en/audio_load)
- [Process audio data](https://huggingface.co/docs/datasets/v2.2.1/en/audio_process)
- [Load and Explore](https://huggingface.co/learn/audio-course/en/chapter1/load_and_explore)
- [Pre-trained ASR Models](https://huggingface.co/learn/audio-course/chapter5/asr_models)

`pip install datasets[audio]`

`python -c "import soundfile; print(soundfile.__libsndfile_version__)"`

> To support loading audio datasets containing MP3 files, users should additionally install `torchaudio`, so that audio data is handled with high performance.

`pip install torchaudio`

> torchaudio’s `sox_io` backend supports decoding mp3 files. Unfortunately, the sox_io backend is only available on Linux/macOS, and is not supported by Windows.

> Audio datasets commonly have an `audio` and `path` or `file` column.

> `audio` is the actual audio file that is loaded and resampled on-the-fly upon calling it.

```python
from datasets import load_dataset, load_metric, Audio

common_voice = load_dataset("common_voice", "tr", split="train")

print(common_voice[0]["audio"])
print(common_voice[0]["path"])
```

> When you access an audio file, it is automatically decoded and resampled. Generally, you should query an audio file like: `common_voice[0]["audio"]`. If you query an audio file with `common_voice["audio"][0]` instead, all the audio files in your dataset will be decoded and resampled. This process can take a long time if you have a large dataset.

> The path is useful if you want to load your own audio dataset. In this case, provide a column of audio file paths to [Dataset.cast_column()](https://huggingface.co/docs/datasets/v2.2.1/en/package_reference/main_classes#datasets.Dataset.cast_column):

`my_audio_dataset = my_audio_dataset.cast_column("paths_to_my_audio_files", Audio())`

## Local audio

> You can load your own dataset using the paths to your audio files. Use the `cast_column()` function to take a column of audio file paths, and cast it to the Audio feature:

```python
audio_dataset = Dataset.from_dict({"audio": ["path/to/audio_1", "path/to/audio_2", ..., "path/to/audio_n"]}).cast_column("audio", Audio())
print(audio_dataset[0]["audio"])
```



### Resample

> Some models expect the audio data to have a certain sampling rate due to how the model was pretrained. For example, the `XLSR-Wav2Vec2` model expects the input to have a sampling rate of 16kHz, but an audio file from the Common Voice dataset has a sampling rate of 48kHz. You can use [Dataset.cast_column()](https://huggingface.co/docs/datasets/v2.2.1/en/package_reference/main_classes#datasets.Dataset.cast_column) to downsample the sampling rate to 16kHz:

`common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16_000))`

![Resample animation](docs/resample.gif)

### Map

> Just like text datasets, you can apply a preprocessing function over an entire dataset with [Dataset.map()](https://huggingface.co/docs/datasets/v2.2.1/en/package_reference/main_classes#datasets.Dataset.map), which is useful for preprocessing all of your audio data at once.

### Fine-tuning

For pretrained speech recognition models, such as `facebook/wav2vec2-large-xlsr-53`, a tokenizer needs to be created from the target text as explained [here](https://huggingface.co/blog/fine-tune-wav2vec2-english). The following example demonstrates how to load a feature extractor, tokenizer and processor for a pretrained speech recognition model.

```python
from transformers import AutoTokenizer, AutoFeatureExtractor, Wav2Vec2Processor
model_checkpoint = "facebook/wav2vec2-large-xlsr-53"
# after defining a vocab.json file you can instantiate a tokenizer object:
tokenizer = AutoTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
processor = Wav2Vec2Processor.from_pretrained(feature_extractor=feature_extractor, tokenizer=tokenizer)
```

For fine-tuned speech recognition models, you can simply load a predefined processor object with:
  
```
from transformers import Wav2Vec2Processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
```

> Make sure to include the audio key in your preprocessing function when you call `Dataset.map()` so that you are actually resampling the audio data:

```python
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch

common_voice_train = common_voice_train.map(prepare_dataset, remove_columns=common_voice_train.column_names)
```

## Vision

`pip install datasets[vision]`

## Pipelines

`pip install -i https://pypi.org/simple/ bitsandbytes`

> If the model is too large for a single GPU and you are using PyTorch, you can set device_map="auto" to automatically determine how to load and store the model weights. Using the device_map argument requires the 🤗 Accelerate package.

## Stream
