experiments_lstm_1024 = [
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
    # {"name": "[M6 / 2º variação]", "frames": 30,
    #     "command": "python3.11 src/main.py --use-velocity --lstm-units 1024 --batch-size 512 --executions 5"},
    # {"name": "[M6 / 3º variação]", "frames": 30,
    #     "command": "python3.11 src/main.py --use-velocity --lstm-units 1024 --batch-size 256 --executions 5"},
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

    # ----------- Tabela 6.6 -----------
    {"name": "[M14 + E2]", "frames": 30,
        "command": "python3.11 src/main.py --use-velocity --use-specialist-16-17 --specialist-only-velocity --lstm-units 1024 --batch-size 1024 --executions 5"},
    {"name": "[M15 + E6]", "frames": 30,
        "command": "python3.11 src/main.py --use-velocity --use-specialist-4-7 --specialist-only-velocity --lstm-units 1024 --batch-size 1024 --executions 5"},
    {"name": "[M16 + E2 + E6]", "frames": 30,
        "command": "python3.11 src/main.py --use-velocity --use-specialist-4-7 --use-specialist-16-17 --specialist-only-velocity --lstm-units 1024 --batch-size 1024 --executions 5"},
]

experiments_batch_size_512 = [
# ----------- Tabela 6.2 -----------
    {"name": "[M2 - Original]", "frames": 30,
        "command": "python3.11 src/main.py --legacy-features --lstm-units 1024 --batch-size 512 --executions 5"},
    {"name": "[M2]", "frames": 30,
        "command": "python3.11 src/main.py --legacy-features --lstm-units 1024 --batch-size 512 --executions 5"},
    {"name": "[M1]", "frames": 30,
        "command": "python3.11 src/main.py --legacy-features --lstm-units 512 --batch-size 512 --executions 5"},
    {"name": "[M3]", "frames": 30,
        "command": "python3.11 src/main.py --legacy-features --lstm-units 256 --batch-size 512 --executions 5"},
    {"name": "[M4]", "frames": 15,
        "command": "python3.11 src/main.py --legacy-features --lstm-units 1024 --batch-size 512 --executions 5"},
    {"name": "[Base]", "frames": 15,
        "command": "python3.11 src/main.py --legacy-features --lstm-units 512 --batch-size 512 --executions 5"},
    {"name": "[M5]", "frames": 15,
        "command": "python3.11 src/main.py --legacy-features --lstm-units 256 --batch-size 512 --executions 5"},

    # ----------- Tabela 6.3 -----------
    {"name": "[M6]", "frames": 30,
        "command": "python3.11 src/main.py --use-velocity --lstm-units 1024 --batch-size 512 --executions 5"},
    {"name": "[M7]", "frames": 30,
        "command": "python3.11 src/main.py --legacy-features --use-velocity --lstm-units 1024 --batch-size 512 --executions 5"},
    {"name": "[M8]", "frames": 30,
        "command": "python3.11 src/main.py --lstm-units 1024 --batch-size 512 --executions 5"},
    {"name": "[M9]", "frames": 30,
        "command": "python3.11 src/main.py --general-only-expansion --use-velocity --lstm-units 1024 --batch-size 512 --executions 5"},
    {"name": "[M10]", "frames": 30,
        "command": "python3.11 src/main.py --general-only-expansion --lstm-units 1024 --batch-size 512 --executions 5"},
    {"name": "[M11]", "frames": 30,
        "command": "python3.11 src/main.py --general-only-velocity --lstm-units 1024 --batch-size 512 --executions 5"},

    # ----------- Tabela 6.4 -----------
    {"name": "[E1]", "frames": 30, "command": "python3.11 src/main.py --train-specialist-only 4 --legacy-features --lstm-units 1024 --batch-size 512 --executions 5"},
    {"name": "[E2]", "frames": 30, "command": "python3.11 src/main.py --train-specialist-only 4 --use-velocity --specialist-only-velocity --lstm-units 1024 --batch-size 512 --executions 5"},
    {"name": "[E3]", "frames": 30,
        "command": "python3.11 src/main.py --train-specialist-only 4 --lstm-units 1024 --batch-size 512 --executions 5"},
    {"name": "[E4]", "frames": 30, "command": "python3.11 src/main.py --train-specialist-only 4 --use-velocity --lstm-units 1024 --batch-size 512 --executions 5"},

    # ----------- Tabela 6.5 -----------
    {"name": "[E5]", "frames": 30, "command": "python3.11 src/main.py --train-specialist-only 16 --legacy-features --lstm-units 1024 --batch-size 512 --executions 5"},
    {"name": "[E6]", "frames": 30, "command": "python3.11 src/main.py --train-specialist-only 16 --use-velocity --specialist-only-velocity --lstm-units 1024 --batch-size 512 --executions 5"},
    {"name": "[E7]", "frames": 30,
        "command": "python3.11 src/main.py --train-specialist-only 16 --lstm-units 1024 --batch-size 512 --executions 5"},
    {"name": "[E8]", "frames": 30,
        "command": "python3.11 src/main.py --train-specialist-only 16 --use-velocity --lstm-units 1024 --batch-size 512 --executions 5"},

    # ----------- Tabela 6.6 -----------
    {"name": "[M14 + E2]", "frames": 30,
        "command": "python3.11 src/main.py --use-velocity --use-specialist-16-17 --specialist-only-velocity --lstm-units 1024 --batch-size 512 --executions 5"},
    {"name": "[M15 + E6]", "frames": 30,
        "command": "python3.11 src/main.py --use-velocity --use-specialist-4-7 --specialist-only-velocity --lstm-units 1024 --batch-size 512 --executions 5"},
    {"name": "[M16 + E2 + E6]", "frames": 30,
        "command": "python3.11 src/main.py --use-velocity --use-specialist-4-7 --use-specialist-16-17 --specialist-only-velocity --lstm-units 1024 --batch-size 512 --executions 5"},
]

experiments_batch_size_256 = [
# ----------- Tabela 6.2 -----------
    {"name": "[M2 - Original]", "frames": 30,
        "command": "python3.11 src/main.py --legacy-features --lstm-units 1024 --batch-size 256 --executions 5"},
    {"name": "[M2]", "frames": 30,
        "command": "python3.11 src/main.py --legacy-features --lstm-units 1024 --batch-size 256 --executions 5"},
    {"name": "[M1]", "frames": 30,
        "command": "python3.11 src/main.py --legacy-features --lstm-units 512 --batch-size 256 --executions 5"},
    {"name": "[M3]", "frames": 30,
        "command": "python3.11 src/main.py --legacy-features --lstm-units 256 --batch-size 256 --executions 5"},
    {"name": "[M4]", "frames": 15,
        "command": "python3.11 src/main.py --legacy-features --lstm-units 1024 --batch-size 256 --executions 5"},
    {"name": "[Base]", "frames": 15,
        "command": "python3.11 src/main.py --legacy-features --lstm-units 512 --batch-size 256 --executions 5"},
    {"name": "[M5]", "frames": 15,
        "command": "python3.11 src/main.py --legacy-features --lstm-units 256 --batch-size 256 --executions 5"},

    # ----------- Tabela 6.3 -----------
    {"name": "[M6]", "frames": 30,
        "command": "python3.11 src/main.py --use-velocity --lstm-units 1024 --batch-size 256 --executions 5"},
    {"name": "[M7]", "frames": 30,
        "command": "python3.11 src/main.py --legacy-features --use-velocity --lstm-units 1024 --batch-size 256 --executions 5"},
    {"name": "[M8]", "frames": 30,
        "command": "python3.11 src/main.py --lstm-units 1024 --batch-size 256 --executions 5"},
    {"name": "[M9]", "frames": 30,
        "command": "python3.11 src/main.py --general-only-expansion --use-velocity --lstm-units 1024 --batch-size 256 --executions 5"},
    {"name": "[M10]", "frames": 30,
        "command": "python3.11 src/main.py --general-only-expansion --lstm-units 1024 --batch-size 256 --executions 5"},
    {"name": "[M11]", "frames": 30,
        "command": "python3.11 src/main.py --general-only-velocity --lstm-units 1024 --batch-size 256 --executions 5"},

    # ----------- Tabela 6.4 -----------
    {"name": "[E1]", "frames": 30, "command": "python3.11 src/main.py --train-specialist-only 4 --legacy-features --lstm-units 1024 --batch-size 256 --executions 5"},
    {"name": "[E2]", "frames": 30, "command": "python3.11 src/main.py --train-specialist-only 4 --use-velocity --specialist-only-velocity --lstm-units 1024 --batch-size 256 --executions 5"},
    {"name": "[E3]", "frames": 30,
        "command": "python3.11 src/main.py --train-specialist-only 4 --lstm-units 1024 --batch-size 256 --executions 5"},
    {"name": "[E4]", "frames": 30, "command": "python3.11 src/main.py --train-specialist-only 4 --use-velocity --lstm-units 1024 --batch-size 256 --executions 5"},

    # ----------- Tabela 6.5 -----------
    {"name": "[E5]", "frames": 30, "command": "python3.11 src/main.py --train-specialist-only 16 --legacy-features --lstm-units 1024 --batch-size 256 --executions 5"},
    {"name": "[E6]", "frames": 30, "command": "python3.11 src/main.py --train-specialist-only 16 --use-velocity --specialist-only-velocity --lstm-units 1024 --batch-size 256 --executions 5"},
    {"name": "[E7]", "frames": 30,
        "command": "python3.11 src/main.py --train-specialist-only 16 --lstm-units 1024 --batch-size 256 --executions 5"},
    {"name": "[E8]", "frames": 30,
        "command": "python3.11 src/main.py --train-specialist-only 16 --use-velocity --lstm-units 1024 --batch-size 256 --executions 5"},

    # ----------- Tabela 6.6 -----------
    {"name": "[M14 + E2]", "frames": 30,
        "command": "python3.11 src/main.py --use-velocity --use-specialist-16-17 --specialist-only-velocity --lstm-units 1024 --batch-size 256 --executions 5"},
    {"name": "[M15 + E6]", "frames": 30,
        "command": "python3.11 src/main.py --use-velocity --use-specialist-4-7 --specialist-only-velocity --lstm-units 1024 --batch-size 256 --executions 5"},
    {"name": "[M16 + E2 + E6]", "frames": 30,
        "command": "python3.11 src/main.py --use-velocity --use-specialist-4-7 --use-specialist-16-17 --specialist-only-velocity --lstm-units 1024 --batch-size 256 --executions 5"},
]

experiments_options = [
  {
    "name": "LSTM 1024",
    "experiments": experiments_lstm_1024
  },
  {
    "name": "Batch Size 512",
    "experiments": experiments_batch_size_512
  },
  {
    "name": "Batch Size 256",
    "experiments": experiments_batch_size_256
  }
]