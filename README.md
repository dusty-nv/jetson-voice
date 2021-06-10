# jetson-voice

jetson-voice is an ASR/NLP/TTS deep learning inference library for Jetson Nano, TX1/TX2, Xavier NX, and AGX Xavier.  It supports Python and JetPack 4.4.1 or newer.  The DNN models were trained with [NeMo](https://github.com/NVIDIA/NeMo) and deployed with [TensorRT](https://developer.nvidia.com/tensorrt) for optimized performance.  All computation is performed using the onboard GPU.

Currently the following capabilities are included:

* [Automatic Speech Recognition (ASR)](#asr)
	* [Streaming ASR (QuartzNet)](#asr) 
	* [Command/Keyword Recognition (MatchboxNet)](#commandkeyword-recognition)
	* [Voice Activity Detection (VAD Marblenet)](#voice-activity-detection-vad)
* [Natural Language Processing (NLP)](#nlp)
	* [Joint Intent/Slot Classification](#joint-intentslot-classification)
	* [Text Classification (Sentiment Analysis)](#text-classification)
	* [Token Classification (Named Entity Recognition)](#token-classification)
	* [Question/Answering (QA)](#questionanswering)
* [Text-to-Speech (TTS)](#tts)
	
The NLP models are using the [DistilBERT](https://arxiv.org/abs/1910.01108) transformer architecture for reduced memory usage and increased performance.  For samples of the text-to-speech output, see the [TTS Audio Samples](#tts-audio-samples) section below.

## Running the Container

jetson-voice is distributed as a Docker container due to the number of dependencies.  There are pre-built containers images available on DockerHub for JetPack 4.4.1 and JetPack 4.5/4.5.1:

```
dustynv/jetson-voice:r32.4.4    # JetPack 4.4.1 (L4T R32.4.4)
dustynv/jetson-voice:r32.5.0    # JetPack 4.5 (L4T R32.5.0) / JetPack 4.5.1 (L4T R32.5.1)
```

To download and run the container, you can simply clone this repo and use the `docker/run.sh` script:

``` bash
$ git clone --branch dev https://github.com/dusty-nv/jetson-voice
$ cd jetson-voice
$ docker/run.sh
```

> **note**:  if you want to use a USB microphone or speaker, plug it in *before* you start the container

There are some optional arguments to `docker/run.sh` that you can use:

* `-r` (`--run`) specifies a run command, otherwise the container will start in an interactive shell.
* `-v` (`--volume`) mount a directory from the host into the container (`/host/path:/container/path`)
* `--dev` starts the container in development mode, where all the source files are mounted for easy editing

The run script will automatically mount the `data/` directory into the container, which stores the models and other data files.  If you save files from the container there, they will also show up under `data/` on the host.

## ASR

The speech recognition in jetson-voice is a streaming service, so it's intended to be used on live sources and transcribes the audio in 1-second chunks.  It uses a [QuartzNet-15x5](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/models.html#quartznet) model followed by a CTC beamsearch decoder and language model, to further refine the raw output of the network.  It detects breaks in the audio to determine the end of sentences.  For information about using the ASR APIs, please refer to [`jetson_voice/asr.py`](jetson_voice/asr.py) and see the [`examples/asr.py`](examples/asr.py)

After you start the container, first run a test audio file (wav/ogg/flac) through [`examples/asr.py`](examples/asr.py) to verify that the system is functional.  Run this command (and all subsequent commands) inside the container:

``` bash
$ examples/asr.py --wav data/audio/dusty.wav

hi
hi hi this is dust
hi hi this is dusty check
hi hi this is dusty check one two
hi hi this is dusty check one two three
hi hi this is dusty check one two three.

what's the weather or
what's the weather going to be tomorrow
what's the weather going to be tomorrow in pittsburgh
what's the weather going to be tomorrow in pittsburgh.

today is
today is wednesday
today is wednesday tomorrow is thursday
today is wednesday tomorrow is thursday.

i would like
i would like to order a large
i would like to order a large pepperoni pizza
i would like to order a large pepperoni pizza.

is it going to be
is it going to be cloudy tomorrow.
```

> The first time you run each model, TensorRT will take a few minutes to optimize it.  
> This optimized model is then cached to disk, so the next time you run the model it will load faster.

#### Live Microphone

To test the ASR on a mic, first list the audio devices in your system to get the audio device ID's:

``` bash
$ scripts/list_audio_devices.sh

----------------------------------------------------
 Audio Input Devices
----------------------------------------------------
Input Device ID 1 - 'tegra-snd-t210ref-mobile-rt565x: - (hw:1,0)' (inputs=16) (sample_rate=44100)
Input Device ID 2 - 'tegra-snd-t210ref-mobile-rt565x: - (hw:1,1)' (inputs=16) (sample_rate=44100)
Input Device ID 3 - 'tegra-snd-t210ref-mobile-rt565x: - (hw:1,2)' (inputs=16) (sample_rate=44100)
Input Device ID 4 - 'tegra-snd-t210ref-mobile-rt565x: - (hw:1,3)' (inputs=16) (sample_rate=44100)
Input Device ID 5 - 'tegra-snd-t210ref-mobile-rt565x: - (hw:1,4)' (inputs=16) (sample_rate=44100)
Input Device ID 6 - 'tegra-snd-t210ref-mobile-rt565x: - (hw:1,5)' (inputs=16) (sample_rate=44100)
Input Device ID 7 - 'tegra-snd-t210ref-mobile-rt565x: - (hw:1,6)' (inputs=16) (sample_rate=44100)
Input Device ID 8 - 'tegra-snd-t210ref-mobile-rt565x: - (hw:1,7)' (inputs=16) (sample_rate=44100)
Input Device ID 9 - 'tegra-snd-t210ref-mobile-rt565x: - (hw:1,8)' (inputs=16) (sample_rate=44100)
Input Device ID 10 - 'tegra-snd-t210ref-mobile-rt565x: - (hw:1,9)' (inputs=16) (sample_rate=44100)
Input Device ID 11 - 'Logitech H570e Mono: USB Audio (hw:2,0)' (inputs=2) (sample_rate=44100)
Input Device ID 12 - 'Samson Meteor Mic: USB Audio (hw:3,0)' (inputs=2) (sample_rate=44100)
```

> If you don't see your audio device listed, exit and restart the container.  
> USB devices should be attached *before* the container is started.

Then run the ASR example with the `--mic <DEVICE>` option, and specify either the device ID or name:

``` bash
$ examples/asr.py --mic 11

hey
hey how are you guys
hey how are you guys.

# (Press Ctrl+C to exit)
```

## ASR Classification

There are other ASR models included for command/keyword recognition ([MatchboxNet](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/speech_classification/models.html#matchboxnet-speech-commands)) and voice activity detection ([VAD MarbleNet](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/speech_classification/models.html#marblenet-vad)).  These models are smaller and faster, and classify chunks of audio as opposed to transcribing text.  

### Command/Keyword Recognition

The [MatchboxNet](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/speech_classification/models.html#matchboxnet-speech-commands) model was trained on 12 keywords from the [Google Speech Commands](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html) dataset:

```
# MatchboxNet classes
"yes",
"no",
"up",
"down",
"left",
"right",
"on",
"off",
"stop",
"go",
"unknown",
"silence"
```

You can run it through the same ASR example as above by specifying the `--model matchboxnet` argument:

``` bash
$ examples/asr.py --model matchboxnet --wav data/audio/commands.wav

class 'unknown' (0.384)
class 'yes' (1.000)
class 'no' (1.000)
class 'up' (1.000)
class 'down' (1.000)
class 'left' (1.000)
class 'left' (1.000)
class 'right' (1.000)
class 'on' (1.000)
class 'off' (1.000)
class 'stop' (1.000)
class 'go' (1.000)
class 'go' (1.000)
class 'silence' (0.639)
class 'silence' (0.576)
```

The numbers printed on the right are the classification probabilities between 0 and 1.

### Voice Activity Detection (VAD)

The voice activity model ([VAD MarbleNet](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/speech_classification/models.html#marblenet-vad)) is a binary model that outputs `background` or `speech`:

``` bash
$ examples/asr.py --model vad_marblenet --wav data/audio/commands.wav

class 'background' (0.969)
class 'background' (0.984)
class 'background' (0.987)
class 'speech' (0.997)
class 'speech' (1.000)
class 'speech' (1.000)
class 'speech' (0.998)
class 'background' (0.987)
class 'speech' (1.000)
class 'speech' (1.000)
class 'speech' (1.000)
class 'background' (0.988)
class 'background' (0.784)
```

## NLP

There are two samples included for NLP:

* [`examples/nlp.py`](examples/nlp.py) (intent/slot, text classification, token classification)
* [`examples/nlp_qa.py`](examples/nlp_qa.py) (question/answering)

These each use a [DistilBERT](https://arxiv.org/abs/1910.01108) model which has been fined-tuned for it's particular task.  For information about using the NLP APIs, please refer to [`jetson_voice/nlp.py`](jetson_voice/nlp.py) and see the samples above.

### Joint Intent/Slot Classification

Joint Intent and Slot classification is a task of classifying an Intent and detecting all relevant Slots (Entities) for this Intent in a query. For example, in the query: `What is the weather in Santa Clara tomorrow morning?`, we would like to classify the query as a `weather` Intent, and detect `Santa Clara` as a location slot and `tomorrow morning` as a date_time slot. 

Intents and Slots names are usually task specific and defined as labels in the training data.  The included intent/slot model was trained on the [NLU-Evaluation-Data](https://github.com/xliuhw/NLU-Evaluation-Data) dataset - you can find the various intent and slot classes that it supports [here](https://gist.github.com/dusty-nv/119474dfcf3bfccfbb8428951a64cd23).  They are common things that you might ask a virtual assistant:

```
$ examples/nlp.py --model distilbert_intent

Enter intent_slot query, or Q to quit:

> What is the weather in Santa Clara tomorrow morning?

{'intent': 'weather_query',
 'score': 0.7165476,
 'slots': [{'score': 0.6280392, 'slot': 'place_name', 'text': 'Santa'},
           {'score': 0.61760694, 'slot': 'place_name', 'text': 'Clara'},
           {'score': 0.5439486, 'slot': 'date', 'text': 'tomorrow'},
           {'score': 0.4520608, 'slot': 'date', 'text': 'morning'}]}

> Set an alarm for 730am

{'intent': 'alarm_set',
 'score': 0.5713072,
 'slots': [{'score': 0.40017933, 'slot': 'time', 'text': '730am'}]}

> Turn up the volume

{'intent': 'audio_volume_up', 'score': 0.33523008, 'slots': []}

> What is my schedule for tomorrow?

{'intent': 'calendar_query',
 'score': 0.37434494,
 'slots': [{'score': 0.5732627, 'slot': 'date', 'text': 'tomorrow'}]}

> order a pepperoni pizza from domino's

{'intent': 'takeaway_order',
 'score': 0.50629586,
 'slots': [{'score': 0.27558547, 'slot': 'food_type', 'text': 'pepperoni'},
           {'score': 0.2778827, 'slot': 'food_type', 'text': 'pizza'},
           {'score': 0.21785143, 'slot': 'business_name', 'text': 'dominos'}]}
```

### Text Classification

In this text classification example, we'll use the included sentiment analysis model that was trained on the [Standford Sentiment Treebank (SST-2)](https://nlp.stanford.edu/sentiment/index.html) dataset.  It will label queries as either positive or negative, along with their probability:

```
$ examples/nlp.py --model distilbert_sentiment

Enter text_classification query, or Q to quit:

> today was warm, sunny and beautiful out

{'class': 1, 'label': '1', 'score': 0.9985898}

> today was cold and rainy and not very nice

{'class': 0, 'label': '0', 'score': 0.99136007}
```

(class 0 is negative sentiment and class 1 is positive sentiment)

### Token Classification

Whereas text classification classifies entire queries, token classification classifies individual tokens (or words).  In this example, we'll be performing Named Entity Recognition (NER), which is the task of detecting and classifying key information (entities) in text. For example, in a sentence: `Mary lives in Santa Clara and works at NVIDIA`, we should detect that `Mary` is a person, `Santa Clara` is a location and `NVIDIA` is a company.

The included token classification model for NER was trained on the [Groningen Meaning Bank (GMB)](http://www.let.rug.nl/bjerva/gmb/about.php) and supports the following annotations in [IOB format](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)) (short for inside, outside, beginning)

* LOC = Geographical Entity
* ORG = Organization
* PER = Person
* GPE = Geopolitical Entity
* TIME = Time indicator
* MISC = Artifact, Event, or Natural Phenomenon

``` bash
$ examples/nlp.py --model distilbert_ner

Enter token_classification query, or Q to quit:
> Mary lives in Santa Clara and works at NVIDIA

Mary[B-PER 0.989] lives in Santa[B-LOC 0.998] Clara[I-LOC 0.996] and works at NVIDIA[B-ORG 0.967]

> Lisa's favorite place to climb in the summer is El Capitan in Yosemite National Park in California, U.S.

Lisa's[B-PER 0.995] favorite place to climb in the summer[B-TIME 0.996] is El[B-PER 0.577] Capitan[I-PER 0.483] 
in Yosemite[B-LOC 0.987] National[I-LOC 0.988] Park[I-LOC 0.98] in California[B-LOC 0.998], U.S[B-LOC 0.997].
```

### Question/Answering

Question/Answering (QA) works by supplying a context paragraph which the model then queries the best answer from.  The [`nlp_qa.py`](examples/nlp_qa.py) example allows you to select from several built-in context paragraphs (or supply your own) and to ask questions about these topics.  

The QA model is flexible and doesn't need re-trained on different topics, as it was trained on the [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) question/answering dataset which allows it to extract answers from a variety of contexts.  It essentially learns to identify the information most relevant to your query from the context passage, as opposed to learning the content itself.

``` bash
$ examples/nlp_qa.py 

Context:
The Amazon rainforest is a moist broadleaf forest that covers most of the Amazon basin of South America. 
This basin encompasses 7,000,000 square kilometres (2,700,000 sq mi), of which 5,500,000 square kilometres 
(2,100,000 sq mi) are covered by the rainforest. The majority of the forest is contained within Brazil, 
with 60% of the rainforest, followed by Peru with 13%, and Colombia with 10%.

Enter a question, C to change context, P to print context, or Q to quit:

> How big is the Amazon?

Answer: 7,000,000 square kilometres
Score:  0.24993503093719482

> which country has the most?

Answer: Brazil
Score:  0.5964332222938538
```

To change the topic or create one of your own, enter `C`:

```
Enter a question, C to change context, P to print context, or Q to quit:
> C

Select from one of the following topics, or enter your own context paragraph:
   1. Amazon
   2. Geology
   3. Moon Landing
   4. Pi
   5. Super Bowl 55
> 3

Context:
The first manned Moon landing was Apollo 11 on July, 20 1969. The first human to step on the Moon was 
astronaut Neil Armstrong followed second by Buzz Aldrin. They landed in the Sea of Tranquility with their 
lunar module the Eagle. They were on the lunar surface for 2.25 hours and collected 50 pounds of moon rocks.

Enter a question, C to change context, P to print context, or Q to quit:

> Who was the first man on the moon?

Answer: Neil Armstrong
Score:  0.39105066657066345
```

## TTS

The text-to-speech service uses an ensemble of two models:  FastPitch to generate MEL spectrograms from text, and HiFiGAN as the vocoder (female English voice).  For information about using the TTS APIs, please refer to [`jetson_voice/tts.py`](jetson_voice/tts.py) and see [`examples/tts.py`](examples/tts.py).

The [`examples/tts.py`](examples/tts.py) app can output the audio to a speaker, wav file, or sequence of wav files.  Run it with `--list-devices` to get a list of your audio devices.

``` bash
$ examples/tts.py --output-device 11 --output-wav data/audio/tts_test

> The weather tomorrow is forecast to be warm and sunny with a high of 83 degrees.

Run 0 -- Time to first audio: 1.820s. Generated 5.36s of audio. RTFx=2.95.
Run 1 -- Time to first audio: 0.232s. Generated 5.36s of audio. RTFx=23.15.
Run 2 -- Time to first audio: 0.230s. Generated 5.36s of audio. RTFx=23.31.
Run 3 -- Time to first audio: 0.231s. Generated 5.36s of audio. RTFx=23.25.
Run 4 -- Time to first audio: 0.230s. Generated 5.36s of audio. RTFx=23.36.
Run 5 -- Time to first audio: 0.230s. Generated 5.36s of audio. RTFx=23.35.

Wrote audio to data/audio/tts_test/0.wav

Enter text, or Q to quit:
> Sally sells seashells by the seashore.

Run 0 -- Time to first audio: 0.316s. Generated 2.73s of audio. RTFx=8.63.
Run 1 -- Time to first audio: 0.126s. Generated 2.73s of audio. RTFx=21.61.
Run 2 -- Time to first audio: 0.127s. Generated 2.73s of audio. RTFx=21.51.
Run 3 -- Time to first audio: 0.126s. Generated 2.73s of audio. RTFx=21.68.
Run 4 -- Time to first audio: 0.126s. Generated 2.73s of audio. RTFx=21.68.
Run 5 -- Time to first audio: 0.126s. Generated 2.73s of audio. RTFx=21.61.

Wrote audio to data/audio/tts_test/1.wav
```

#### TTS Audio Samples

* [Weather forecast](data/audio/tts_examples/0.wav) (wav)
* [Sally sells seashells](data/audio/tts_examples/1.wav) (wav)


## Tests

There is an automated test suite included that will verify all of the models are working properly.  You can run it with the `tests/run_tests.py` script:

``` bash
$ tests/run_tests.py

----------------------------------------------------
 TEST SUMMARY
----------------------------------------------------
test_asr.py (quartznet)                  PASSED
test_asr.py (quartznet_greedy)           PASSED
test_asr.py (matchboxnet)                PASSED
test_asr.py (vad_marblenet)              PASSED
test_nlp.py (distilbert_qa_128)          PASSED
test_nlp.py (distilbert_qa_384)          PASSED
test_nlp.py (distilbert_intent)          PASSED
test_nlp.py (distilbert_sentiment)       PASSED
test_nlp.py (distilbert_ner)             PASSED
test_tts.py (fastpitch_hifigan)          PASSED

passed 10 of 10 tests
saved logs to data/tests/logs/20210610_1512
```

The logs of the individual tests are printed to the screen and saved to a timestamped directory.



