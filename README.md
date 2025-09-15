# MIDI Music Maker

The Midi Music Maker will complete music for you. Just give it a few notes and it will generate the next few notes. 
It will take in input of midi music and run it via a MIDI LLM transformer to output the next few notes via midi.

## How this works

This project takes a MIDI input from a keyboard or other midi device and process it via an LLM. 

Each LLM I have trained is a tranfomer model based on Mistral architecture. These have been trained from scratch. 
The tokens are encoded using a custom MidiTok Remi encoding. 

[Single instrument trained model](https://huggingface.co/adricl/midi_single_instrument_mistral_transformer)

[Multi instrument trained model](https://huggingface.co/adricl/music_transformer_playground)


## Linux installation requirements.
python-dev
libasound2-dev

To setup the audio
sudo usermod -a -G audio $USER

## Logic for the processing of midi input.

1. Callback reads the MIDI input from the device and puts it into a buffer
2. Once there is a pause of the input for 5 seconds we then copy the buffer and write it to a file then process it via the LLM.
3. The LLM will then ouptut the next few notes.

## Running the code
````
pip install -r requirements.txt
````

example command line where v4/ contains the model files and v4/.json is the tokenizer ReMOTE is the input midi device and FLUID is the output midi device
````
python midi_music_maker.py v4/ v4/HuggingFace_Mistral_Transformer_Single_Instrument_v4_single_track.json v4/ -imp ReMOTE -omp FLUID
````
If you want to use the pipe input use -f and you can specify the input pipe file


## Setting up fluidsynth for playing back midi output
````
sudo apt-get install fluidsynth
sudo apt-get install fluid-soundfont-gm
````
This skips starting with Jack. We assume that .sf2 is your soundsfont file.

````
fluidsynth FatBoy-v0.786.sf2  -a pulseaudio -m alsa_seq -o midi.autoconnect=1
````