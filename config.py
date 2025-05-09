# Paths
BASE_DIR = r"D:\thesis.code\paper-krol-modular\data\AFDB"               # Folder where your original records are stored
OUTPUT_DIR = r"D:\thesis.code\paper-krol-modular\data\scalogram"       # Where to save scalograms & metadata

# ECG Parameters
AFIB_LABEL = 1
SEGMENT_DURATION_SEC = 5
FS = 250
SEGMENT_LENGTH = SEGMENT_DURATION_SEC * FS

# Record list
RECORDS = [
    "04015", "04048", "04126", "04746", "04908",
    "05121", "05261", "06426", "06995", "07162",
    "07859", "07879", "07910", "08215", "08219"
]
