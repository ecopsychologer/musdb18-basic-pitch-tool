import os, gc, tqdm, musdb, stempeg
from pathlib import Path
from basic_pitch.inference import predict_and_save

def safe_make_directory(path):
    """Safely create a directory if it doesn't already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)

def load_processed_tracks(log_path):
    """Load processed tracks from the log file."""
    if not os.path.exists(log_path):
        return set()
    with open(log_path, 'r') as log_file:
        return set(log_file.read().splitlines())

def log_processed_track(log_path, track_name):
    """Log a newly processed track."""
    with open(log_path, 'a') as log_file:
        log_file.write(f"{track_name}\n")

def save_stem_and_transcribe(audio_data, stem_name, track_name, save_dir, rate, processed_tracks, log_path):
    """Save a single stem audio to the disk and transcribe it to MIDI."""
    filename_base = f"{track_name.replace('/', '-')}_{stem_name}"
    stem_path = Path(save_dir, stem_name, f"{filename_base}.wav")

    # Check if this stem has already been processed
    if filename_base in processed_tracks:
        print(f"Skipping already processed track: {filename_base}")
        return
    
    # Ensure directory exists
    safe_make_directory(stem_path.parent)
    
    # Save stem
    stempeg.write_audio(
        path=str(stem_path),
        data=audio_data,
        sample_rate=rate
    )
    
    # Ensure MIDI output directory exists
    midi_output_dir = Path(save_dir, stem_name, "midi")
    safe_make_directory(midi_output_dir)
    
    print(f"Transcribing to MIDI: {stem_path}")  # Debugging print
    try:
        predict_and_save(
            [str(stem_path)],  # Correctly pass the list of audio file paths
            output_directory=str(midi_output_dir),  # Output directory for the MIDI files
            save_midi=True,
            sonify_midi=False,
            save_model_outputs=False,
            save_notes=False
        )
        log_processed_track(log_path, filename_base)  # Log successful processing
    except Exception as e:
        print(f"Error during transcription: {e}")
        
    gc.collect()

def process_musdb_dataset(musdb_path, save_dir):
    """Process and save stems from the MUSDB18 dataset."""
    mus = musdb.DB(root=musdb_path, is_wav=False)
    print(f"Found {len(mus.tracks)} tracks in the dataset.")
    
    log_path = "processed_tracks.log"  # Path to your log file
    processed_tracks = load_processed_tracks(log_path)
    
    stem_names = ['mixture', 'drums', 'bass', 'other', 'vocals']
    for name in stem_names:
        safe_make_directory(os.path.join(save_dir, name))
    
    total_tracks = len(mus.tracks)
    for i, track in enumerate(tqdm.tqdm(mus.tracks, desc="Processing tracks"), 1):
        print(f"Processing {track.name} [{i}/{total_tracks}]...")
        for stem_name in stem_names:
            if stem_name in track.targets:
                save_stem_and_transcribe(
                    track.targets[stem_name].audio,
                    stem_name,
                    track.name,
                    save_dir,
                    track.rate,
                    processed_tracks,
                    log_path
                )
        print(f"Finished processing {track.name}")

if __name__ == "__main__":
    musdb_path = "../training_data/musdb18"  # Adjust path as necessary
    save_dir = "../training_data/stems"  # Adjust path as necessary
    process_musdb_dataset(musdb_path, save_dir)
