specialists_only = [
    # ----------- Tabela 6.4 -----------
    {"name": "[E1]", "frames": 30, "command": "python3.11 src/main.py --train-specialist-only 4 --legacy-features --lstm-units 1024 --batch-size 1024 --executions 5"},
    {"name": "[E2]", "frames": 30, "command": "python3.11 src/main.py --train-specialist-only 4 --use-velocity --specialist-only-velocity --lstm-units 1024 --batch-size 1024 --executions 5"},
    {"name": "[E3]", "frames": 30,
        "command": "python3.11 src/main.py --train-specialist-only 4 --lstm-units 1024 --batch-size 1024 --executions 5"},
    {"name": "[E4]", "frames": 30, "command": "python3.11 src/main.py --train-specialist-only 4 --use-velocity --lstm-units 1024 --batch-size 1024 --executions 5"},

    # ----------- Tabela 6.5 -----------
    {"name": "[E5]", "frames": 30, "command": "python3.11 src/main.py --train-specialist-only 16 --legacy-features --lstm-units 1024 --batch-size 1024 --executions 5"},
    {"name": "[E6]", "frames": 30, "command": "python3.11 src/main.py --train-specialist-only 16 --use-velocity --specialist-only-velocity --lstm-units 1024 --batch-size 1024 --executions 5"},
    {"name": "[E7]", "frames": 30,
        "command": "python3.11 src/main.py --train-specialist-only 16 --lstm-units 1024 --batch-size 1024 --executions 5"},
    {"name": "[E8]", "frames": 30,
        "command": "python3.11 src/main.py --train-specialist-only 16 --use-velocity --lstm-units 1024 --batch-size 1024 --executions 5"},
]


def generate_experiments(lstm_units: int, batch_size: int, executions: int):
    return [
        # ----------- Tabela 6.2 -----------
        {"name": "[M2]", "frames": 30,
            "command": f"python3.11 src/main.py --legacy-features --lstm-units 1024 --batch-size {batch_size} --executions {executions}"},
        {"name": "[M1]", "frames": 30,
            "command": f"python3.11 src/main.py --legacy-features --lstm-units 512 --batch-size {batch_size} --executions {executions}"},
        {"name": "[M3]", "frames": 30,
            "command": f"python3.11 src/main.py --legacy-features --lstm-units 256 --batch-size {batch_size} --executions {executions}"},
        {"name": "[M4]", "frames": 15,
            "command": f"python3.11 src/main.py --legacy-features --lstm-units 1024 --batch-size {batch_size} --executions {executions}"},
        {"name": "[Base]", "frames": 15,
            "command": f"python3.11 src/main.py --legacy-features --lstm-units 512 --batch-size {batch_size} --executions {executions}"},
        {"name": "[M5]", "frames": 15,
            "command": f"python3.11 src/main.py --legacy-features --lstm-units 256 --batch-size {batch_size} --executions {executions}"},

        # ----------- Tabela 6.3 -----------
        {"name": "[M6]", "frames": 30,
            "command": f"python3.11 src/main.py --use-velocity --lstm-units {lstm_units} --batch-size {batch_size} --executions {executions}"},
        {"name": "[M7]", "frames": 30,
            "command": f"python3.11 src/main.py --legacy-features --use-velocity --lstm-units {lstm_units} --batch-size {batch_size} --executions {executions}"},
        {"name": "[M8]", "frames": 30,
            "command": f"python3.11 src/main.py --lstm-units {lstm_units} --batch-size {batch_size} --executions {executions}"},
        {"name": "[M9]", "frames": 30,
            "command": f"python3.11 src/main.py --general-only-expansion --use-velocity --lstm-units {lstm_units} --batch-size {batch_size} --executions {executions}"},
        {"name": "[M10]", "frames": 30,
            "command": f"python3.11 src/main.py --general-only-expansion --lstm-units {lstm_units} --batch-size {batch_size} --executions {executions}"},
        {"name": "[M11]", "frames": 30,
            "command": f"python3.11 src/main.py --general-only-velocity --lstm-units {lstm_units} --batch-size {batch_size} --executions {executions}"},

        # ----------- Tabela 6.6 -----------
        {"name": "[M14 + E2]", "frames": 30,
            "command": f"python3.11 src/main.py --use-velocity --use-specialist-16-17 --specialist-only-velocity --lstm-units {lstm_units} --batch-size {batch_size} --executions {executions}"},
        {"name": "[M15 + E6]", "frames": 30,
            "command": f"python3.11 src/main.py --use-velocity --use-specialist-4-7 --specialist-only-velocity --lstm-units {lstm_units} --batch-size {batch_size} --executions {executions}"},
        {"name": "[M16 + E2 + E6]", "frames": 30,
            "command": f"python3.11 src/main.py --use-velocity --use-specialist-4-7 --use-specialist-16-17 --specialist-only-velocity --lstm-units {lstm_units} --batch-size {batch_size} --executions {executions}"},
    ]


experiments_options = [
    {
        "name": "Specialists only",
        "experiments": specialists_only
    },
    {
        "name": "Batch Size 2048",
        "experiments": generate_experiments(1024, 2048, 5)
    },
    {
        "name": "Batch Size 1024",
        "experiments": generate_experiments(1024, 1024, 5)
    },
    {
        "name": "Batch Size 512",
        "experiments": generate_experiments(1024, 512, 5)
    },
    {
        "name": "Batch Size 256",
        "experiments": generate_experiments(1024, 256, 5)
    },
    {
        "name": "Batch Size 128",
        "experiments": generate_experiments(1024, 128, 5)
    },
    {
        "name": "Batch Size 64",
        "experiments": generate_experiments(1024, 64, 5)
    },
    {
        "name": "Batch Size 32",
        "experiments": generate_experiments(1024, 32, 5)
    },
    {
        "name": "Batch Size 16",
        "experiments": generate_experiments(1024, 16, 5)
    },
    {
        "name": "Extra 256 + Batch Size 128",
        "experiments": [
            {"name": "[M16 + E2 + E6] - Variação (Exp + Mov)", "frames": 30,
                        "command": f"python3.11 src/main.py --general-only-expansion --use-velocity --use-specialist-4-7 --use-specialist-16-17 --specialist-only-velocity --lstm-units 1024 --batch-size 256 --executions 5"},
            *generate_experiments(1024, 128, 5)
        ]
    },
]