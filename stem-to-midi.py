import os
import musdb
import numpy as np
from basic_pitch.inference import predict_and_save

def safe_make_directory(path):
    """Safely create a directory if it doesn't already exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def save_stem_and_transcribe(stem_audio, stem_name, track_name, save_dir):
    """Save a single stem audio to the disk and transcribe it to MIDI."""
    filename_base = f"{track_name.replace('/', '-')}_{stem_name}"
    stem_path = os.path.join(save_dir, stem_name, f"{filename_base}.wav")
    musdb.audio.save_wav(stem_path, stem_audio)
    
    # Transcribe to MIDI
    midi_output_dir = os.path.join(save_dir, stem_name, "midi")
    safe_make_directory(midi_output_dir)
    predict_and_save([stem_path], midi_output_dir, save_midi=True)

def process_musdb_dataset(musdb_path, save_dir):
    """Process and save stems from the MUSDB18 dataset."""
    # Load the MUSDB18 dataset
    mus = musdb.DB(root=musdb_path, is_wav=False)
    print(f"Found {len(mus.tracks)} tracks in the dataset.")

    
    # Define stem names based on the MUSDB18 convention
    stem_names = ['mixture', 'drums', 'bass', 'other', 'vocals']
    
    # Ensure directories for each stem exist
    for name in stem_names:
        safe_make_directory(os.path.join(save_dir, name))
    
    # Iterate over tracks in the dataset
    total_tracks = len(mus.tracks)
    for i, track in enumerate(mus.tracks, 1):
        print(f"Processing {track.name} [{i}/{total_tracks}]...")
        stems = track.stems
        # Save each stem individually and transcribe to MIDI
        for stem_name in stem_names:
            save_stem_and_transcribe(track.targets[stem_name].audio, stem_name, track.name, save_dir)
        print(f"Finished processing {track.name}")

if __name__ == "__main__":
    musdb_path = "../training_data/musdb18"  # MUSDB18 dataset root folder
    save_dir = "../training_data/stems"  # The directory where you want to save the stems and MIDI files
    process_musdb_dataset(musdb_path, save_dir)
