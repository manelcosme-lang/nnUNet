import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

# O nnU-Net exige nomes no formato DatasetXXX_Nome; o 001 é o ID usado nos comandos (-d 1)
DATASET_NAME = "Dataset001_Heart"

parser = argparse.ArgumentParser(
    description="Organiza os dados de CT do MM-WHS para o nnU-Net.")
parser.add_argument("pasta_dados", type=Path,
                    help="pasta com os ficheiros ct_train_XXXX_image/label.nii.gz")
parser.add_argument("--teste", type=Path, default=None,
                    help="pasta com os ficheiros ct_test_XXXX_image.nii.gz (opcional)")
parser.add_argument("--saida", type=Path, default=None,
                    help="pasta nnUNet_raw onde criar o dataset (por defeito: variável nnUNet_raw)")
args = parser.parse_args()

source_dir = args.pasta_dados.expanduser().resolve()

env_raw = os.environ.get("nnUNet_raw")
if args.saida is not None:
    nnunet_raw = args.saida.expanduser().resolve()
elif env_raw:
    nnunet_raw = Path(env_raw).expanduser().resolve()
else:
    sys.exit("[ERRO] Não sei onde criar o dataset.\n"
             "Indica a pasta com --saida ou define a variável nnUNet_raw.")

if not source_dir.is_dir():
    sys.exit(f"[ERRO] Pasta de dados não encontrada: {source_dir}")

if args.teste is not None and not args.teste.expanduser().is_dir():
    sys.exit(f"[ERRO] Pasta de teste não encontrada: {args.teste}")


# =============================================================================
# PARTE 1: organizar os dados de treino (o antigo organize_dataset.py)
# =============================================================================

# Padrão dos ficheiros originais: ct_train_1001_image.nii.gz / ct_train_1001_label.nii.gz
pattern = re.compile(r"ct_train_(\d+)_(image|label)\.nii\.gz")

cases = {}  # case_id -> {"image": path, "label": path}

for f in sorted(source_dir.iterdir()):
    match = pattern.match(f.name)
    if not match:
        print(f"[AVISO] Ficheiro ignorado (não corresponde ao padrão esperado): {f.name}")
        continue
    case_id, kind = match.groups()
    cases.setdefault(case_id, {})[kind] = f

# Verificar pares completos
complete_cases = []
for case_id, files in sorted(cases.items()):
    if "image" in files and "label" in files:
        complete_cases.append(case_id)
    else:
        print(f"[AVISO] Caso {case_id} incompleto (falta {'image' if 'image' not in files else 'label'}) — ignorado")

if not complete_cases:
    sys.exit(f"[ERRO] Nenhum par ct_train_XXXX_image/label.nii.gz encontrado em {source_dir}")

print(f"\nTotal de casos completos encontrados: {len(complete_cases)}\n")

dataset_dir = nnunet_raw / DATASET_NAME
images_tr = dataset_dir / "imagesTr"
labels_tr = dataset_dir / "labelsTr"

images_tr.mkdir(parents=True, exist_ok=True)
labels_tr.mkdir(parents=True, exist_ok=True)

# Copiar e renomear
for case_id in complete_cases:
    new_name = f"heart_{case_id}"
    src_image = cases[case_id]["image"]
    src_label = cases[case_id]["label"]

    dst_image = images_tr / f"{new_name}_0000.nii.gz"
    dst_label = labels_tr / f"{new_name}.nii.gz"

    shutil.copy2(src_image, dst_image)
    shutil.copy2(src_label, dst_label)
    print(f"  {src_image.name} -> {dst_image.name}")
    print(f"  {src_label.name} -> {dst_label.name}")

# Criar dataset.json
# As classes correspondem aos valores depois do remapeamento (PARTE 3)
dataset_json = {
    "channel_names": {
        "0": "CT"
    },
    "labels": {
        "background": 0,
        "myocardium_lv": 1,
        "left_atrium": 2,
        "left_ventricle": 3,
        "right_atrium": 4,
        "right_ventricle": 5,
        "ascending_aorta": 6,
        "pulmonary_artery": 7
    },
    "numTraining": len(complete_cases),
    "file_ending": ".nii.gz"
}

with open(dataset_dir / "dataset.json", "w") as f:
    json.dump(dataset_json, f, indent=4)

print(f"\ndataset.json criado em: {dataset_dir / 'dataset.json'}")
print(f"Estrutura pronta em: {dataset_dir}")


# =============================================================================
# PARTE 2: organizar os dados de teste (NOVO)
# Mesma lógica da PARTE 1, para os ficheiros ct_test_XXXX. As imagens vão para
# imagesTs; os labels de teste, se existirem, vão para labelsTs.
# =============================================================================

labels_ts = dataset_dir / "labelsTs"

if args.teste is not None:
    test_dir = args.teste.expanduser().resolve()

    test_pattern = re.compile(r"ct_test_(\d+)_(image|label)\.nii\.gz")

    test_cases = {}  # case_id -> {"image": path, "label": path}
    for f in sorted(test_dir.iterdir()):
        match = test_pattern.match(f.name)
        if match:
            case_id, kind = match.groups()
            test_cases.setdefault(case_id, {})[kind] = f

    if not any("image" in files for files in test_cases.values()):
        sys.exit(f"[ERRO] Nenhum ficheiro ct_test_XXXX_image.nii.gz encontrado em {test_dir}")

    images_ts = dataset_dir / "imagesTs"
    images_ts.mkdir(exist_ok=True)

    print("")
    for case_id, files in sorted(test_cases.items()):
        if "image" not in files:
            continue
        new_name = f"heart_{case_id}"
        shutil.copy2(files["image"], images_ts / f"{new_name}_0000.nii.gz")
        print(f"  {files['image'].name} -> {new_name}_0000.nii.gz")
        if "label" in files:
            labels_ts.mkdir(exist_ok=True)
            shutil.copy2(files["label"], labels_ts / f"{new_name}.nii.gz")
            print(f"  {files['label'].name} -> {new_name}.nii.gz")


# =============================================================================
# PARTE 3: remapear os labels (o antigo remap_labels.py)
# Troca os códigos do MM-WHS (205, 420, ...) pelos valores seguidos 0 a 7 que o
# nnU-Net exige. Os códigos foram escolhidos pelos autores do dataset e não são
# intensidades; a troca não muda a forma das estruturas.
# Corre sobre as cópias feitas nas partes 1 e 2: os ficheiros originais não são alterados.
# =============================================================================

MAPPING = {0: 0, 205: 1, 420: 2, 500: 3, 550: 4, 600: 5, 820: 6, 850: 7}
NEW_VALUES = set(MAPPING.values())

labels_dirs = [labels_tr]
if labels_ts.is_dir():
    labels_dirs.append(labels_ts)

for labels_dir in labels_dirs:
    label_files = sorted(labels_dir.glob("*.nii.gz"))
    print(f"\nEncontrados {len(label_files)} ficheiros de label em {labels_dir}\n")

    for f in label_files:
        img = sitk.ReadImage(str(f))
        arr = sitk.GetArrayFromImage(img)
        values = set(np.unique(arr).tolist())

        # Se o ficheiro já só tem valores de 0 a 7, já foi remapeado: não mexer
        if values <= NEW_VALUES:
            print(f"  {f.name} -> já remapeado, ignorado")
            continue

        remapped = np.zeros_like(arr, dtype=np.uint8)
        for old_val, new_val in MAPPING.items():
            remapped[arr == old_val] = new_val
        unmapped = values - set(MAPPING.keys())
        if unmapped:
            print(f"[AVISO] {f.name}: valores nao mapeados (ficam como fundo): {unmapped}")
        new_img = sitk.GetImageFromArray(remapped)
        new_img.CopyInformation(img)  # mantém espaçamento, origem e orientação do original
        sitk.WriteImage(new_img, str(f))
        print(f"  {f.name} -> remapeado")

print("\nConcluido.")

# O nnU-Net procura os dados na variável nnUNet_raw, por isso tem de apontar para a mesma pasta
if not env_raw or Path(env_raw).expanduser().resolve() != nnunet_raw:
    print("\n[AVISO] A variável nnUNet_raw não aponta para esta pasta. Antes de correres o nnU-Net:")
    print(f'    export nnUNet_raw="{nnunet_raw}"')

print("\nPróximo passo: nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity")