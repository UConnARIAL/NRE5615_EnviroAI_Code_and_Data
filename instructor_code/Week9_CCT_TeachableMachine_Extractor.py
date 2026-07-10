# Week 9 Camera Trap Dataset Extractor
# Creates a small balanced image dataset for Google Teachable Machine.
# Classes: animal, empty, human_vehicle

from pathlib import Path
import json
import random
import shutil
from collections import defaultdict

# -----------------------------
# Update these paths
# -----------------------------
IMAGE_ROOT = Path('/content/cct_images')
ANNOTATION_JSON = Path('/content/annotations.json')
OUTPUT_ROOT = Path('/content/Week9_CameraTrap_TeachableMachine')

# Sample size per class
N_TRAIN = 60
N_TEST = 15

random.seed(42)


def map_to_teaching_class(category_names):
    """
    Convert original camera-trap labels into three simple teaching classes:
    animal, empty, human_vehicle.
    """
    labels = [str(x).lower() for x in category_names]

    if any(x in labels for x in ['empty', 'blank']):
        return 'empty'

    human_vehicle_terms = [
        'human', 'person', 'people',
        'vehicle', 'car', 'truck', 'bike', 'motorcycle'
    ]

    if any(x in labels for x in human_vehicle_terms):
        return 'human_vehicle'

    return 'animal'


def main():
    with open(ANNOTATION_JSON, 'r') as f:
        data = json.load(f)

    images = data['images']
    annotations = data['annotations']
    categories = data['categories']

    print('Number of images:', len(images))
    print('Number of annotations:', len(annotations))
    print('Number of categories:', len(categories))

    cat_id_to_name = {c['id']: c['name'].lower() for c in categories}
    image_id_to_file = {img['id']: img['file_name'] for img in images}

    image_id_to_categories = defaultdict(list)
    for ann in annotations:
        image_id = ann['image_id']
        category_id = ann['category_id']
        category_name = cat_id_to_name.get(category_id, 'unknown')
        image_id_to_categories[image_id].append(category_name)

    print('\nExample categories:')
    for item in list(cat_id_to_name.items())[:10]:
        print(item)

    class_to_files = defaultdict(list)
    missing_files = 0

    for image_id, file_name in image_id_to_file.items():
        category_names = image_id_to_categories.get(image_id, ['empty'])
        teaching_class = map_to_teaching_class(category_names)
        image_path = IMAGE_ROOT / file_name

        if image_path.exists():
            class_to_files[teaching_class].append(image_path)
        else:
            missing_files += 1

    print('\nMissing files:', missing_files)
    print('Available images by teaching class:')
    for cls in ['animal', 'empty', 'human_vehicle']:
        print(f'  {cls}: {len(class_to_files[cls])}')

    needed_per_class = N_TRAIN + N_TEST
    sampled = {}

    for cls in ['animal', 'empty', 'human_vehicle']:
        files = class_to_files[cls]

        if len(files) < needed_per_class:
            print(f"Warning: class '{cls}' has only {len(files)} images. Using all available images.")
            selected = files.copy()
        else:
            selected = random.sample(files, needed_per_class)

        train_files = selected[:N_TRAIN]
        test_files = selected[N_TRAIN:N_TRAIN + N_TEST]

        sampled[cls] = {'train': train_files, 'test': test_files}

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    for split in ['train', 'test']:
        for cls in ['animal', 'empty', 'human_vehicle']:
            out_dir = OUTPUT_ROOT / split / cls
            out_dir.mkdir(parents=True, exist_ok=True)

            for i, src_path in enumerate(sampled[cls][split], start=1):
                suffix = src_path.suffix.lower()
                new_name = f'{cls}_{split}_{i:03d}{suffix}'
                dst_path = out_dir / new_name
                shutil.copy2(src_path, dst_path)

    print('\nFinal folder counts:')
    for split in ['train', 'test']:
        print(split.upper())
        for cls in ['animal', 'empty', 'human_vehicle']:
            folder = OUTPUT_ROOT / split / cls
            count = len(list(folder.glob('*')))
            print(f'  {cls}: {count}')

    zip_path = shutil.make_archive(
        base_name=str(OUTPUT_ROOT),
        format='zip',
        root_dir=OUTPUT_ROOT.parent,
        base_dir=OUTPUT_ROOT.name
    )

    print('\nCreated zip file:')
    print(zip_path)


if __name__ == '__main__':
    main()
