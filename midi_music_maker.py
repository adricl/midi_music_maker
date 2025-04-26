###
# MIDI Music Maker
# The Midi Music Maker will complete music for you. 
# Just give it a few notes and it will generate the next few notes. 
# It will take in input of midi music and run it via a MIDI LLM transformer to 
# output the next few notes via midi.

### What else needs to be done
# Convert to midi input and output using the midi cables
# with mido.open_input(port_name) as inport:
# sudo apt install libasound2-dev
# sudo usermod -a -G audio $USER

import os
import errno
import time
from pathlib import Path
from io import BytesIO
import argparse
import torch
import mido
from symusic import Score
from miditok import REMI
from transformers import AutoModelForCausalLM, GenerationConfig, AutoConfig

# Commandline Args python midi_music_maker.py "/home/wombat/Documents/projects/music/midiTok/data/HuggingFace_Mistral_Transformer_Single_Instrument/run" "/media/wombat/c6928dc9-ba03-411d-9483-8e28df5973b9/Music Data/HuggingFace_Mistral_Transformer_Single_Instrument/HuggingFace_Mistral_Transformer_Single_Instrument.json" "/home/wombat/Documents/projects/music/hf_music_transformer_playground"

def process_midi(model, inp, generation_config, tokenizer, save_path="./bloop.mid"):
    
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

    res = model.generate(
        inputs=tensor_sequence,
        generation_config=generation_config)
    
    print("Generated Output Shape", res.shape)

    decoded = tokenizer.decode([res[0]])
    decoded.dump_midi(save_path)
    print('Duration: {}'.format(time.time() - start_time))

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
                        process_midi(model, data, generation_config, tokenizer, save_path=args.save_path)
    
    except KeyboardInterrupt:
        print("Stopping...")
        os.unlink(pipe_name)
        print("Pipe removed")
        exit(0)

def find_midi_device():
    print("Searching for MIDI devices...")
    input_ports = mido.get_input_names()
    
    for port in input_ports:
        print(f"Found MIDI input port: {port}")
        if "USB" in port:
            print(f"Using MIDI input port: {port}")
            return port
        
    print("No suitable MIDI input device found.")
    exit(1)

def midi_input_to_midi():
    print("Waiting for midi input...")
    
    port_name = find_midi_device()
    
    try:
        with mido.open_input(port_name) as inport:
            print("MIDI device connected! Start playing...")
            
            # Create a MIDI file to collect messages
            midi_file = mido.MidiFile(ticks_per_beat=480, type=0)
            track = mido.MidiTrack()
            midi_file.tracks.append(track)
            
            # Recording state
            recording = False
            last_message_time = time.time()
            silence_threshold = 2.0  # seconds of silence to trigger processing
            
            # Listen for messages
            while True: #TODO Fix this loop 
                msg = inport.receive(block=False)
                current_time = time.time()
                
                if msg is not None:
                    print(f"Received: {msg}")
                    last_message_time = current_time
                    
                    # Start recording on first note_on
                    if msg.type == 'note_on' and msg.velocity > 0:
                        if not recording:
                            recording = True
                            print("Recording started...")
                            track = mido.MidiTrack()
                            midi_file.tracks = [track]
                    
                    # Add message to track if we're recording
                    if recording:
                        track.append(msg)
                
                # Process after silence
                if recording and (current_time - last_message_time) > silence_threshold:
                    print("Silence detected, processing recording...")
                    recording = False
                    
                    # Save to temporary file
                    file = BytesIO()
                    file.name = "temp.mid"
                   
                    midi_file.save(file)
                    
                    # Process the MIDI data
                    process_midi(model, file, generation_config, tokenizer, save_path=args.save_path)
                    
                    # Clean up
                    os.close(file)
                
                # Prevent high CPU usage
                time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("MIDI input monitoring stopped.")
    except Exception as e:
        print(f"Error reading MIDI input: {e}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="generate_realtime.py",
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

    print(args.path_to_model)
    path_path_model = Path(args.path_to_model)

    config_path = str(path_path_model / "config.json")
    print("config_path", config_path)

    model_path = str(path_path_model / "model.safetensors")
    print("model_path", model_path)
    
    config = AutoConfig.from_pretrained(config_path)
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_path, from_tf=False, config=config)
    tokenizer = REMI(params=Path(args.tokeniser_file))
    print(model)
    print(model.config)


    generation_config = GenerationConfig( # This should be in the model config
        max_new_tokens=200,  # extends samples by 200 tokens
        num_beams=1,         # no beam search
        do_sample=True,      # but sample instead
        temperature=0.9,
        top_k=15,
        top_p=0.95,
        epsilon_cutoff=3e-4,
        eta_cutoff=1e-3,
        pad_token_id=tokenizer['PAD_None'], #maybe a good idea but I am not sure    
        bos_token_id=tokenizer['BOS_None'],
        eos_token_id=tokenizer['EOS_None'],
    )

    if (args.pipe_input):
        print("Pipe input mode")
        pipe_to_midi(pipe_name)
    else:
        midi_input_to_midi()


    