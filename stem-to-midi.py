import os, gc, tqdm, musdb, stempeg
from pathlib import Path
from basic_pitch.inference import predict_and_save

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


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

def update_log_from_existing_midi_files(output_dir, log_path):
    """Scan the output directory for existing MIDI files and update the log."""
    processed_tracks = set()
    midi_files = Path(output_dir).rglob("*.mid")
    
    for midi_file in midi_files:
        # Extract track and stem name from the MIDI file path
        # This depends on your directory structure and naming convention
        parts = midi_file.relative_to(output_dir).parts
        if len(parts) >= 3:  # ['stem_name', 'midi', 'track_name_stem_name.midi']
            stem_name = parts[0]
            track_name = parts[2].split('_')[0]
            processed_tracks.add(f"{track_name}_{stem_name}")
            print(f"Logging processed midi stem: {track_name}_{stem_name}")
    
    # Update log file with processed tracks
    with open(log_path, 'w') as log_file:
        for track in processed_tracks:
            log_file.write(f"{track}\n")
            print(f"Successfully logged processed track: {track}")
    
    return processed_tracks


def save_stem_and_transcribe(audio_data, stem_name, track_name, save_dir, rate, log_path):
    """Save a single stem audio to the disk and transcribe it to MIDI."""
    filename_base = f"{track_name.replace('/', '-')}_{stem_name}"
    stem_path = Path(save_dir, stem_name, f"{filename_base}.wav")
    midi_output_dir = Path(save_dir, stem_name, "midi")
    midi_file_path = midi_output_dir / f"{filename_base}_basic_pitch.mid"  # Update to your naming convention

    # Check directly if the MIDI file exists to decide on processing
    if midi_file_path.exists():
        print(f"Skipping already processed track: {filename_base}")
        return

    # Ensure directory exists
    safe_make_directory(stem_path.parent)
    safe_make_directory(midi_output_dir)
    
    # Save stem
    stempeg.write_audio(path=str(stem_path), data=audio_data, sample_rate=rate)
    
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
                    log_path
                )
        print(f"Finished processing {track.name}")

if __name__ == "__main__":
    musdb_path = "../training_data/musdb18"  # Adjust path as necessary
    save_dir = "../training_data/stems"  # Adjust path as necessary
    process_musdb_dataset(musdb_path, save_dir)
