import os
import shutil
import random
import argparse


def load_patch_id_wsi(path):
    f = open(path)
    lines = [line.strip().split(',') for line in f][1:]
    patch_name_to_id_wsi = {}
    for patch_name, id_wsi in lines:
        patch_name_to_id_wsi[patch_name] = id_wsi
    return patch_name_to_id_wsi


def load_data_list(path):
    return [name.split('.')[0] for name in os.listdir(path)]


def make_batch_lists(patch_names_without_id_wsi, patch_name_to_id_wsi, batch_count):
    required_batch_size = (len(patch_names_without_id_wsi) + len(patch_name_to_id_wsi)) // batch_count

    id_wsi_to_patch_names = {}
    for patch_name, id_wsi in patch_name_to_id_wsi.items():
        if id_wsi not in id_wsi_to_patch_names:
            id_wsi_to_patch_names[id_wsi] = []
        id_wsi_to_patch_names[id_wsi].append(patch_name)

    batch_lists = [[] for _ in range(batch_count)]

    for id_wsi, patch_names in id_wsi_to_patch_names.items():
        for i in range(len(batch_lists)):
            if len(batch_lists[i]) + len(patch_names) < required_batch_size:
                batch_lists[i] += patch_names
                break

    random.shuffle(patch_names_without_id_wsi)

    for i in range(len(batch_lists)):
        if len(batch_lists[i]) < required_batch_size:
            if i == len(batch_lists) - 1:
                add_count = len(patch_names_without_id_wsi)
            else:
                add_count = required_batch_size - len(batch_lists[i])
            batch_lists[i] += patch_names_without_id_wsi[:add_count]
            del patch_names_without_id_wsi[:add_count]

    return batch_lists


def make_batches(data_path, path, batch_lists):
    for i in range(len(batch_lists)):
        print("Batch {}: {}".format(i + 1, len(batch_lists[i])))
        batch_path = os.path.join(path, "batch_{}".format(i + 1))
        os.mkdir(os.path.join(path, batch_path))
        for name in batch_lists[i]:
            image_name = "{}.tif".format(name)
            shutil.copyfile(os.path.join(data_path, image_name), os.path.join(batch_path, image_name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Path to folder with train data")
    parser.add_argument("--batch_path", required=True, help="Path to folder with batches")
    parser.add_argument("--batch_count", type=int, required=True, help="Count of batches")
    parser.add_argument("--patch_id_wsi_path", required=True, help="Path to csv file with path id")
    parser.add_argument("--random_seed", type=int, default=42, help="Seed for random numbers")
    args = parser.parse_args()

    print("Loading data list...")
    data_list = load_data_list(args.data_path)
    print("Dataset size: {}".format(len(data_list)))

    print("Batch count: {}".format(args.batch_count))
    required_batch_size = len(data_list) // args.batch_count
    print("Required batch size: {}".format(required_batch_size))

    print("Loading wsi ids")
    patch_name_to_id_wsi = load_patch_id_wsi(args.patch_id_wsi_path)
    print("Loaded {} patches with wsi id".format(len(patch_name_to_id_wsi)))

    patch_names_without_id_wsi = []
    for patch_name in data_list:
        if patch_name not in patch_name_to_id_wsi:
            patch_names_without_id_wsi.append(patch_name)
    print("Loaded {} patches without wsi id".format(len(patch_names_without_id_wsi)))

    batch_lists = make_batch_lists(patch_names_without_id_wsi, patch_name_to_id_wsi, args.batch_count)

    make_batches(args.data_path, args.batch_path, batch_lists)


if __name__ == "__main__":
    main()
