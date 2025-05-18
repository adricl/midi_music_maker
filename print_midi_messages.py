import argparse
import mido

def main():
    parser = argparse.ArgumentParser(description="Print all MIDI messages from a MIDI file.")
    parser.add_argument("midi_file", help="Path to the MIDI file to read")
    args = parser.parse_args()

    try:
        mid = mido.MidiFile(args.midi_file)
        for i, msg in enumerate(mid):
            print(f"Message {i}: {msg}")
    except Exception as e:
        print(f"Error reading MIDI file: {e}")

if __name__ == "__main__":
    main()