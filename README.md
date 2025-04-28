# MIDI Music Maker

The Midi Music Maker will complete music for you. Just give it a few notes and it will generate the next few notes. 
It will take in input of midi music and run it via a MIDI LLM transformer to output the next few notes via midi.

## How this works.

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
