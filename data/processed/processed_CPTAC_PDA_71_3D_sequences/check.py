import os


def get_wsi_ids_from_directory(directory):
    """
    Estrae gli ID dai nomi dei file/cartelle nella directory fornita.
    """
    return {
        name.split("-")[0] + "-" + name.split("-")[1] for name in os.listdir(directory)
    }


def get_ct_ids_from_directory(directory):
    """
    Estrae gli ID dai nomi dei file/cartelle nella directory fornita.
    """
    return {name for name in os.listdir(directory)}


def find_matching_ids(ct_dir, wsi_dir, label_file):
    """
    Trova gli ID comuni tra le due directory e li stampa.
    """
    ct_ids = get_ct_ids_from_directory(ct_dir)
    wsi_ids = get_wsi_ids_from_directory(wsi_dir)
    labels = load_labels(label_file)
    matching_ids = ct_ids.intersection(wsi_ids)
    for match in matching_ids:
        label = labels.get(match, "Unknown")
        print(f"{match}: {label}")


def load_labels(label_file):
    """
    Carica le etichette da un file di testo e le restituisce come dizionario.
    """
    labels = {}
    with open(label_file) as file:
        for line in file:
            parts = line.strip().split(", ")
            if len(parts) == 2:
                labels[parts[0]] = parts[1]
    return labels


# Esempio di utilizzo
ct_directory = "CT"  # Cambia con il percorso reale
wsi_directory = "WSI"  # Cambia con il percorso reale
label_file = "labels.txt"
find_matching_ids(ct_directory, wsi_directory, label_file)
