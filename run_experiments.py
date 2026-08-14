import re
import subprocess
import time
from datetime import datetime
from experiments import experiments_options

# Path to the Settings.py file
SETTINGS_PATH = "src/entities/Settings.py"


def update_settings(num_frames):
    """Updates the NUM_FRAMES and FEATURES_PATH variables in Settings.py."""
    with open(SETTINGS_PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    # Use regex to find and replace NUM_FRAMES
    new_content = re.sub(r'NUM_FRAMES:\s*int\s*=\s*\d+',
                         f'NUM_FRAMES: int = {num_frames}', content)

    # Use regex to find and replace FEATURES_PATH
    if num_frames == 15:
        features_path = "data/features-hands-distances-normal-face-126"
    else:
        features_path = "data/features-hands-distances-normal-face-126-frames-30"

    new_content = re.sub(r'FEATURES_PATH:\s*str\s*=\s*".*"',
                         f'FEATURES_PATH: str = "{features_path}"', new_content)

    with open(SETTINGS_PATH, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(
        f" -> Settings.py updated: NUM_FRAMES = {num_frames}, FEATURES_PATH = {features_path}")


def run_experiment(name, frames, command):
    print(f"\n{'='*50}")
    print(f"Starting Experiment: {name}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")

    update_settings(frames)

    print(f" -> Command: {command}")

    # Run the command and wait for it to finish
    try:
        # We use shell=True so it can run the full command string
        # check=True raises an exception if the command fails
        subprocess.run(command, shell=True, check=True)
        print(f"\n[SUCCESS] {name} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] {name} failed with return code {e.returncode}.")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error running {name}: {e}")


if __name__ == "__main__":
    for index, option in enumerate(experiments_options):
        print(f"[{index}] - {option.name}")

    experiment_choice = input("Choose the experiments you want to run")

    try:
        experiment_object = experiments_options[int(experiment_choice)]
        experiments = experiment_object.experiments

        print("Starting experiment suite...")
        print(f"Total experiments to run: {len(experiments)}")

        start_time = time.time()

        for i, exp in enumerate(experiments, 1):
            print(f"\n--- Progress: {i}/{len(experiments)} ---")
            run_experiment(exp["name"], exp["frames"], exp["command"])

        end_time = time.time()
        duration = end_time - start_time
        hours, rem = divmod(duration, 3600)
        minutes, seconds = divmod(rem, 60)

        print("\n" + "="*50)
        print("ALL EXPERIMENTS COMPLETED")
        print(
            f"Total time taken: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
        print("="*50)
    except Exception as error:
        print("Error:", error)
