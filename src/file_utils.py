from os import walk, path, makedirs


def list_filepaths_with_extension(dir_path: str, extension: str) -> list[str]:
    """Lista todos os arquivos com a extensão dada nas subpastas do diretório."""
    filepaths = []
    for root, _, files in walk(dir_path):
        for file in files:
            if file.lower().endswith(extension.lower()):
                filepaths.append(path.join(root, file))

    return filepaths

def get_folder_name(video_file: str) -> str:
    return path.basename(path.dirname(video_file))

def get_filename(video_file: str) -> str:
    return path.basename(video_file)

def make_directories(dir_path: str):
    makedirs(dir_path, exist_ok=True)
