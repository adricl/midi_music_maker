###
# MIDI Music Maker
# The Midi Music Maker will complete music for you.
# Just give it a few notes and it will generate the next few notes.
# It will take in input of midi music and run it via a MIDI LLM transformer to
# output the next few notes via midi.

### What else needs to be done
# Convert to midi input and output using the midi cables

import os
import errno
import time
from pathlib import Path
from io import BytesIO
import argparse
import torch
import mido
from mido import MidiFile
from symusic import Score
from miditok import REMI
from transformers import AutoModelForCausalLM, GenerationConfig, AutoConfig

# Commandline Args python midi_music_maker.py "/home/wombat/Documents/projects/music/midiTok/data/HuggingFace_Mistral_Transformer_Single_Instrument/run" "/media/wombat/c6928dc9-ba03-411d-9483-8e28df5973b9/Music Data/HuggingFace_Mistral_Transformer_Single_Instrument/HuggingFace_Mistral_Transformer_Single_Instrument.json" "/home/wombat/Documents/projects/music/hf_music_transformer_playground"

#Global Variables

#Default MIDI file parameters
TICKS_PER_BEAT = 480
DEFAULT_TEMPO = 500000  # Microseconds per beat (120 BPM)

SILENCE_THRESHOLD = 5.0  # seconds of silence to trigger processing


def process_midi(model, inp, generation_config, tokenizer, save_path):

    start_time = time.time()

    score = Score.from_midi(inp)

    print("Processing midi file")
    tokenized_input = tokenizer(score)

    max_len = model.config.max_position_embeddings
    print(f"Max position embeddings: {model.config.max_position_embeddings}")
    max_len = 1024 #TODO for now as we are using a smaller model

    print(f"Tokenized input shape: {len(tokenized_input[0].ids)}")
    input_ids = tokenized_input[0].ids
    if len(input_ids) >= max_len:
        print(f"Warning: Input sequence ({len(input_ids)}) longer than max_position_embeddings ({max_len}). Truncating.")
        input_ids = input_ids[-(max_len - 1):]


    tensor_sequence = torch.tensor([input_ids], dtype=torch.long)
    print(f"Current tensor shape: {tensor_sequence.shape}")
    input_token_length = tensor_sequence.shape[1]

    res = model.generate(
        inputs=tensor_sequence,
        generation_config=generation_config)

    print("Generated Output Shape", res.shape)

    decoded = tokenizer.decode([res[0][input_token_length:]])

    file_path = Path(save_path) / f"{time.time()}_generated.mid"
    decoded.dump_midi(file_path)
    print('Duration: {}'.format(time.time() - start_time))
    print(f"Decoded shape: {decoded}") # Note: `decoded` is a Score object, printing it might be verbose.

    midi_to_output_midi(file_path)


def midi_to_output_midi(midi_file):
    output_port = find_midi_output_device()
    if output_port is None:
        print("No MIDI output device found or specified. Skipping playback.")
        return
    try:
        with mido.open_output(output_port) as outport:
            # Play MIDI file - this uses the time values in the MIDI file
            for msg in MidiFile(midi_file).play():
                type = msg.dict()['type']
                if not (msg.is_meta or type == 'program_change'):  # Skip meta messages and program_change
                    outport.send(msg)
                    print(f"Sent: {msg}")
    except Exception as e:
        print(f"Error sending MIDI to output device: {e}")

def pipe_to_midi(pipe_name):

    try:
        if (os.path.exists(pipe_name)):
            print("Pipe already exists, removing...")
            os.unlink(pipe_name)

        os.mkfifo(pipe_name)
        print("Creating Pipe...")
    except OSError as oe:
        print(oe)
        if oe.errno != errno.EEXIST:
            raise

    try:
        while True:
            with open(pipe_name, mode='rb') as fifo:
                while True:
                    data = fifo.read()
                    if len(data) == 0:
                        print("Waiting for new data")
                        break
                    else:
                        # Ensure global variables are accessible or passed if needed
                        process_midi(model, data, generation_config, tokenizer, save_path=args.save_path)

    except KeyboardInterrupt:
        print("Stopping...")
        os.unlink(pipe_name)
        print("Pipe removed")
        exit(0)

def find_midi_input_device():
    print("Searching for MIDI input devices...")
    input_ports = mido.get_input_names()

    port = find_usb_midi_device(input_ports)
    if port is not None:
        print(f"Found MIDI input port: {port}")
        return port

    print("No suitable MIDI input device found. Please connect a USB MIDI device.")
    exit(1)


def find_midi_output_device():
    print("Searching for MIDI output devices...")
    output_ports = mido.get_output_names()

    port = find_usb_midi_device(output_ports)
    if port is not None:
        print(f"Found MIDI output port: {port}")
        return port

    print("No suitable MIDI output device found. Please connect a USB MIDI device.")
    exit(1)


def find_usb_midi_device(ports):
    for port in ports:
        if "USB" in port: # Check if the port name contains "USB"
            return port
    return None

def start_new_recording():
    midi_file = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = mido.MidiTrack()
    midi_file.tracks.append(track)

    # Add initial tempo message. Its time is 0 as it's the first in this track.
    track.append(mido.MetaMessage('set_tempo', tempo=DEFAULT_TEMPO, time=0))

    return midi_file, track

def save_process_recording(midi_file):
    # Save to temporary BytesIO stream
    mem_file = BytesIO()
    mem_file.name = "temp_recorded.mid"
    midi_file.save(file=mem_file)
    mem_file.seek(0) # Rewind to start of stream for reading

    mem_file_data = mem_file.read() # Read data for process_midi

    # Save the recorded MIDI to a file for inspection/debugging (optional)
    mem_file.seek(0) # Rewind again to read for saving to disk
    played_midi_path = Path(args.save_path) / f"{time.time()}_played.mid"
    with open(played_midi_path, "wb") as f:
        f.write(mem_file.read())
    print(f"Saved played MIDI to: {played_midi_path}")

    print(mem_file_data)

    # Process the MIDI data (pass bytes)
    process_midi(model, mem_file_data, generation_config, tokenizer, save_path=args.save_path)
    print("Processing complete.")
    mem_file.close()

def midi_input_to_midi():
    print("Waiting for midi input...")

    port_name = find_midi_input_device()

    if port_name is None:
        print("Exiting due to no MIDI input device found.")
        exit(1)

    # State variables for recording
    midi_file = None  # Will be mido.MidiFile object when recording
    track = None      # Will be mido.MidiTrack object, part of midi_file
    recording = False
    last_message_time = time.time()  # For silence detection
    time_of_last_track_event_abs = 0.0  # Absolute time of the last event added to the current track
    inport = mido.open_input(port_name)

    try:
        #with mido.open_input(port_name) as inport: #move this
            print(f"MIDI device '{port_name}' connected! Start playing...")

            while True: # Main loop for receiving messages
                if (not inport.is_open):
                    print(f"Reopening MIDI input port: {port_name}")
                    inport = mido.open_input(port_name)

                msg = inport.receive(block=False) # Non-blocking receive
                current_event_time_abs = time.time() # Get time for this potential event

                if msg is not None and msg.type != 'clock': # Ignore clock messages
                    print(f"Received: {msg}")
                    last_message_time = current_event_time_abs # Update for silence detection

                    # Start recording on first note_on message
                    if msg.type == 'note_on' and msg.velocity > 0:
                        if not recording:
                            print("Recording started...")
                            recording = True

                            midi_file, track = start_new_recording()
                            # The time of this first "event" (the tempo message) is current_event_time_abs.
                            # Subsequent actual MIDI messages will be timed relative to this moment.
                            time_of_last_track_event_abs = current_event_time_abs

                    # Add message to track if we're recording
                    if recording:
                        if track is None or midi_file is None:
                            print("Error: Recording is true but track/midi_file is not initialized.")
                            # This state should ideally not be reached if logic is correct.
                        else:
                            message_for_track = msg.copy()
                            # Calculate delta time in seconds since the last event *added to the track*
                            delta_seconds = current_event_time_abs - time_of_last_track_event_abs
                            delta_ticks = mido.second2tick(delta_seconds, midi_file.ticks_per_beat, DEFAULT_TEMPO)

                            # msg.time from inport.receive() is delta from previous *port* message, not what we need here.
                            message_for_track.time = int(round(delta_ticks)) # MIDI ticks must be integers
                            track.append(message_for_track)

                            # Update the absolute time of the last recorded event in the track
                            time_of_last_track_event_abs = current_event_time_abs

                # Process after silence
                if recording and (current_event_time_abs - last_message_time) > SILENCE_THRESHOLD:
                    print("Silence detected, processing recording...")

                    if midi_file: # Ensure midi_file exists (it should if recording was true)
                        inport.close()  # Close the input port to stop receiving messages
                        save_process_recording(midi_file)

                    # Reset state for the next recording segment
                    recording = False
                    midi_file = None  # Clear the file object
                    track = None      # Clear the track object
                    # time_of_last_track_event_abs will be reset when new recording starts.
                    # Reset last_message_time to current time to base silence detection correctly for post-processing period
                    last_message_time = time.time()

    except KeyboardInterrupt:
        inport.close()
        print("MIDI input monitoring stopped.")
    except Exception as e:
        inport.close()
        print(f"Error reading MIDI input: {e}")
        import traceback
        traceback.print_exc()

# Removed create_midi_file function as its logic is now inlined

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="midi_music_maker.py", 
        description="Generate midi audio real time "
    )
    parser.add_argument("path_to_model", help="string path to model directory", type=str)
    parser.add_argument("tokeniser_file", help="path to tokeniser json file", type=str)
    parser.add_argument("save_path", help="path at which to save the generated midi file", type=str)
    parser.add_argument("-v", "--verbose", help="verbose output flag", action="store_true")
    parser.add_argument("-p", "--pipe_name", help="Pipe name", type=str)
    parser.add_argument("-f", "--pipe_input", help="Input Midi from a pipe", action="store_true")

    args = parser.parse_args()

    # fix arguments
    pipe_name = args.pipe_name if args.pipe_name else "music_transfomer_pipe"

    print(f"Model Path: {args.path_to_model}")
    path_path_model = Path(args.path_to_model)

    # It's generally safer to load Hugging Face models and configs directly from the directory path
    # if it's a saved model directory, rather than individual files like config.json or model.safetensors.
    # However, if this structure is required by your setup, it's fine.
    # For from_pretrained, usually the directory containing config.json and model files is enough.
    
    # Assuming path_to_model is the directory containing the model files (config.json, model.safetensors etc.)
    try:
        config = AutoConfig.from_pretrained(args.path_to_model)
        model = AutoModelForCausalLM.from_pretrained(args.path_to_model, config=config)
    except Exception as e:
        print(f"Error loading model from {args.path_to_model}: {e}")
        print("Ensure 'path_to_model' is a directory containing 'config.json' and model files (e.g., 'model.safetensors' or 'pytorch_model.bin').")
        exit(1)
        
    tokenizer = REMI(params=Path(args.tokeniser_file))
    print(model)
    print(model.config)

    generation_config = GenerationConfig(
        max_new_tokens=200,
        num_beams=1,
        do_sample=True,
        temperature=0.9,
        top_k=15,
        top_p=0.95,
        epsilon_cutoff=3e-4,
        eta_cutoff=1e-3,
        pad_token_id=tokenizer['PAD_None'],
        bos_token_id=tokenizer['BOS_None'],
        eos_token_id=tokenizer['EOS_None'],
    )

    if (args.pipe_input):
        print("Pipe input mode")
        pipe_to_midi(pipe_name)
    else:
        midi_input_to_midi()