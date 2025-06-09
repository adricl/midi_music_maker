###
# MIDI Music Maker
# The Midi Music Maker will complete music for you.
# Just give it a few notes and it will generate the next few notes.
# It will take in input of midi music and run it via a MIDI LLM transformer to
# output the next few notes via midi.


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
        input_ids = input_ids[-max_len:]


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
                if not (msg.is_meta or msg.type == 'program_change'):  # Skip meta messages and program_change
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

def create_new_midi_file():
    midi_file = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = mido.MidiTrack()
    midi_file.tracks.append(track)

    # Add initial tempo message. Its time is 0 as it's the first in this track.
    track.append(mido.MetaMessage('set_tempo', tempo=DEFAULT_TEMPO, time=0))

    return midi_file, track

def save_buffer_and_process_recording(message_buffer):

    if message_buffer is None or len(message_buffer) == 0:
        print("No messages to process, skipping...")
        return
    
    midi_file, track = create_new_midi_file()
    # Maybe add code here to get the first message in the buffer that is msg.type == 'note_on' and msg.velocity > 0:
    for msg in message_buffer:
        track.append(msg)

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
    time_of_last_event_in_buffer = 0.0  # Absolute time of the last event added to the current track
    _message_buffer = []  # Buffer for incoming messages Underscore to indicate it's managed by this function and its callback

    def midi_input_callback(msg):
        nonlocal time_of_last_event_in_buffer, _message_buffer
        if msg is not None and msg.type != 'clock': # Ignore clock messages
            
            current_event_time_abs = time.time() # Update for silence detection

            if not _message_buffer or len(_message_buffer) == 0:
                time_of_last_event_in_buffer = current_event_time_abs

            # Start recording on first note_on message
            message_for_track = msg.copy()
            # Calculate delta time in seconds since the last event *added to the track*
            delta_seconds = current_event_time_abs - time_of_last_event_in_buffer
            if delta_seconds < 0: # Should not happen with time.time() but good for robustness
                delta_seconds = 0.0

            delta_ticks = mido.second2tick(delta_seconds, TICKS_PER_BEAT, DEFAULT_TEMPO)

            # msg.time from inport.receive() is delta from previous *port* message, not what we need here.
            message_for_track.time = int(round(delta_ticks)) # MIDI ticks must be integers
            _message_buffer.append(message_for_track)
            print(f"Received: {message_for_track}")
            # Update the absolute time of the last recorded event in the track
            time_of_last_event_in_buffer = current_event_time_abs

    inport = None  # Initialize inport to None to ensure it's defined in the finally block
    try:
        inport = mido.open_input(port_name, callback=midi_input_callback)
        print(f"MIDI device '{port_name}' connected! Start playing...")

        while True:
            if _message_buffer and (time.time() - time_of_last_event_in_buffer) > SILENCE_THRESHOLD:
                buffer = _message_buffer.copy()  # Copy current buffer to process
                _message_buffer.clear()  # Clear the buffer for new messages
                save_buffer_and_process_recording(buffer)
                _message_buffer.clear()  # Clear the buffer after processing so we dont get junk in the next recording TODO: might think of something smarter to do here.

    except KeyboardInterrupt:
        print("MIDI input monitoring stopped.")
    except Exception as e:
        print(f"Error reading MIDI input: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if inport and not inport.closed:
            print(f"Closing MIDI port '{port_name}'.")
            inport.close()


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