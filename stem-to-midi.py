import os
import tqdm
import musdb
import stempeg
from pathlib import Path
from basic_pitch.inference import predict_and_save

def safe_make_directory(path):
    """Safely create a directory if it doesn't already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)

def save_stem_and_transcribe(audio_data, stem_name, track_name, save_dir, rate):
    """Save a single stem audio to the disk and transcribe it to MIDI."""
    filename_base = f"{track_name.replace('/', '-')}_{stem_name}"
    stem_path = Path(save_dir, stem_name, f"{filename_base}.wav")

    # Ensure directory exists
    safe_make_directory(stem_path.parent)
    
    # Save stem
    stempeg.write_audio(
        path=str(stem_path),
        data=audio_data,
        sample_rate=rate
    )
    
    # Transcribe to MIDI
    midi_output_dir = Path(save_dir, stem_name, "midi")
    safe_make_directory(midi_output_dir)
    predict_and_save(str(midi_output_dir),
                     str(stem_path),
                     save_midi=True,
                     sonify_midi=False,
                     save_model_outputs=False,
                     save_notes=False
    )

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
    for i, track in enumerate(tqdm.tqdm(mus.tracks, desc="Processing tracks"), 1):
        print(f"Processing {track.name} [{i}/{total_tracks}]...")
        for stem_name in stem_names:
            # Note: Check if the target exists before processing
            if stem_name in track.targets:
                save_stem_and_transcribe(
                    audio_data=track.targets[stem_name].audio,
                    stem_name=stem_name,
                    track_name=track.name,
                    save_dir=save_dir,
                    rate=track.rate
                )
        print(f"Finished processing {track.name}")

if __name__ == "__main__":
    musdb_path = "../training_data/musdb18"  # MUSDB18 dataset root folder
    save_dir = "../training_data/stems"  # The directory where you want to save the stems and MIDI files
    process_musdb_dataset(musdb_path, save_dir)
