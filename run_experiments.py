import re
import subprocess
import time
from datetime import datetime

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


experiments = [
    # ----------- Tabela 6.2 -----------
    {"name": "[M2 - Original]", "frames": 30,
        "command": "python3.11 src/main.py --legacy-features --lstm-units 1024 --batch-size 512 --executions 5"},
    {"name": "[M2]", "frames": 30,
        "command": "python3.11 src/main.py --legacy-features --lstm-units 1024 --batch-size 1024 --executions 5"},
    {"name": "[M1]", "frames": 30,
        "command": "python3.11 src/main.py --legacy-features --lstm-units 512 --batch-size 1024 --executions 5"},
    {"name": "[M3]", "frames": 30,
        "command": "python3.11 src/main.py --legacy-features --lstm-units 256 --batch-size 1024 --executions 5"},
    {"name": "[M4]", "frames": 15,
        "command": "python3.11 src/main.py --legacy-features --lstm-units 1024 --batch-size 1024 --executions 5"},
    {"name": "[Base]", "frames": 15,
        "command": "python3.11 src/main.py --legacy-features --lstm-units 512 --batch-size 1024 --executions 5"},
    {"name": "[M5]", "frames": 15,
        "command": "python3.11 src/main.py --legacy-features --lstm-units 256 --batch-size 1024 --executions 5"},

    # ----------- Tabela 6.3 -----------
    {"name": "[M6]", "frames": 30,
        "command": "python3.11 src/main.py --use-velocity --lstm-units 1024 --batch-size 1024 --executions 5"},
    {"name": "[M6 / 2º variação]", "frames": 30,
        "command": "python3.11 src/main.py --use-velocity --lstm-units 1024 --batch-size 512 --executions 5"},
    {"name": "[M6 / 3º variação]", "frames": 30,
        "command": "python3.11 src/main.py --use-velocity --lstm-units 1024 --batch-size 256 --executions 5"},
    {"name": "[M7]", "frames": 30,
        "command": "python3.11 src/main.py --legacy-features --use-velocity --lstm-units 1024 --batch-size 1024 --executions 5"},
    {"name": "[M8]", "frames": 30,
        "command": "python3.11 src/main.py --lstm-units 1024 --batch-size 1024 --executions 5"},
    {"name": "[M9]", "frames": 30,
        "command": "python3.11 src/main.py --general-only-expansion --use-velocity --lstm-units 1024 --batch-size 1024 --executions 5"},
    {"name": "[M10]", "frames": 30,
        "command": "python3.11 src/main.py --general-only-expansion --lstm-units 1024 --batch-size 1024 --executions 5"},
    {"name": "[M11]", "frames": 30,
        "command": "python3.11 src/main.py --general-only-velocity --lstm-units 1024 --batch-size 1024 --executions 5"},

    # ----------- Tabela 6.4 -----------
    {"name": "[E1]", "frames": 30, "command": "python3.11 src/main.py --train-specialist-only 16 --legacy-features --lstm-units 1024 --batch-size 1024 --executions 5"},
    {"name": "[E2]", "frames": 30, "command": "python3.11 src/main.py --train-specialist-only 16 --use-velocity --specialist-only-velocity --lstm-units 1024 --batch-size 1024 --executions 5"},
    {"name": "[E3]", "frames": 30,
        "command": "python3.11 src/main.py --train-specialist-only 16 --lstm-units 1024 --batch-size 1024 --executions 5"},
    {"name": "[E4]", "frames": 30, "command": "python3.11 src/main.py --train-specialist-only 16 --use-velocity --lstm-units 1024 --batch-size 1024 --executions 5"},

    # ----------- Tabela 6.5 -----------
    {"name": "[E5]", "frames": 30, "command": "python3.11 src/main.py --train-specialist-only 4 --legacy-features --lstm-units 1024 --batch-size 1024 --executions 5"},
    {"name": "[E6]", "frames": 30, "command": "python3.11 src/main.py --train-specialist-only 4 --use-velocity --specialist-only-velocity --lstm-units 1024 --batch-size 1024 --executions 5"},
    {"name": "[E7]", "frames": 30,
        "command": "python3.11 src/main.py --train-specialist-only 4 --lstm-units 1024 --batch-size 1024 --executions 5"},
    {"name": "[E8]", "frames": 30,
        "command": "python3.11 src/main.py --train-specialist-only 4 --use-velocity --lstm-units 1024 --batch-size 1024 --executions 5"},

    # ----------- Tabela 6.6 -----------
    {"name": "[M14 + E2]", "frames": 30,
        "command": "python3.11 src/main.py --use-velocity --use-specialist-16-17 --specialist-only-velocity --lstm-units 1024 --batch-size 1024 --executions 5"},
    {"name": "[M15 + E6]", "frames": 30,
        "command": "python3.11 src/main.py --use-velocity --use-specialist-4-7 --specialist-only-velocity --lstm-units 1024 --batch-size 1024 --executions 5"},
    {"name": "[M16 + E2 + E6]", "frames": 30,
        "command": "python3.11 src/main.py --use-velocity --use-specialist-4-7 --use-specialist-16-17 --specialist-only-velocity --lstm-units 1024 --batch-size 1024 --executions 5"},
]

if __name__ == "__main__":
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
