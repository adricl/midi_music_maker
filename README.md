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

## Logic for the processing of midi input.

1. Callback reads the MIDI input from the device in a thread that keeps reading it and saving it to a file.
2. Once there is a pause of the input for 5 seconds we then process the file through the LLM on a different thread from the input reading thread.
3. While the processing is occuring we need to change the file we are writing to. Once the processing is done we need to write to another new file. 


Global Variables
last_midi_input_time
midi_input_file
 

Callback 
Records midi inputs. 
For each input we need to set a timestamp variable to the current time.
takes the file it will need to write out to from the global variable.

Main loop,

detect if the last midi input time is more than 5 seconds ago and if it is process the file.
When processing a new file we need to 

