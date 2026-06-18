# Emilia Dataset: https://huggingface.co/datasets/amphion/Emilia-Dataset/tree/fc71e07
# if use updated new version, i.e. WebDataset, feel free to modify / draft your own script
import glob
# generate audio text map for Emilia DE & EN
# evaluate for vocab size

import os
import shutil
import sys

import pandas as pd

from sklearn.model_selection import train_test_split

sys.path.append(os.getcwd())

import json
from concurrent.futures import ProcessPoolExecutor
from importlib.resources import files
from pathlib import Path
from tqdm import tqdm

from datasets.arrow_writer import ArrowWriter

from f5_tts.model.utils import (
    repetition_found,
    convert_char_to_pinyin,
)

cv_out_de = {}
cv_de_filters =  [
    '་', 'Ə', 'Ḫ', 'ů', '`', 'Ș', 'Œ', 'Ī', 'ġ', 'ſ', '‚', '‟', '‹', 'Ř', 'ŏ', '̆', 'Ț', 'ན', 'Ő', 'ḫ', 'ẞ', '̥', 'ṟ', '›',
    'ď', 'Ħ', 'Ѹ', 'Ô', '„',
]
cv_out_en = {}
cv_en_filters = []

emilia_out_de = {}
# emilia (english default) + cv (weird german symbols)
emilia_de_filters = ["ا", "い", "て"]
# seems synthesized audios, or heavily code-switched
emilia_out_en = {
    "EN_B00013_S00913",
    "EN_B00042_S00120",
    "EN_B00055_S04111",
    "EN_B00061_S00693",
    "EN_B00061_S01494",
    "EN_B00061_S03375",
    "EN_B00059_S00092",
    "EN_B00111_S04300",
    "EN_B00100_S03759",
    "EN_B00087_S03811",
    "EN_B00059_S00950",
    "EN_B00089_S00946",
    "EN_B00078_S05127",
    "EN_B00070_S04089",
    "EN_B00074_S09659",
    "EN_B00061_S06983",
    "EN_B00061_S07060",
    "EN_B00059_S08397",
    "EN_B00082_S06192",
    "EN_B00091_S01238",
    "EN_B00089_S07349",
    "EN_B00070_S04343",
    "EN_B00061_S02400",
    "EN_B00076_S01262",
    "EN_B00068_S06467",
    "EN_B00076_S02943",
    "EN_B00064_S05954",
    "EN_B00061_S05386",
    "EN_B00066_S06544",
    "EN_B00076_S06944",
    "EN_B00072_S08620",
    "EN_B00076_S07135",
    "EN_B00076_S09127",
    "EN_B00065_S00497",
    "EN_B00059_S06227",
    "EN_B00063_S02859",
    "EN_B00075_S01547",
    "EN_B00061_S08286",
    "EN_B00079_S02901",
    "EN_B00092_S03643",
    "EN_B00096_S08653",
    "EN_B00063_S04297",
    "EN_B00063_S04614",
    "EN_B00079_S04698",
    "EN_B00104_S01666",
    "EN_B00061_S09504",
    "EN_B00061_S09694",
    "EN_B00065_S05444",
    "EN_B00063_S06860",
    "EN_B00065_S05725",
    "EN_B00069_S07628",
    "EN_B00083_S03875",
    "EN_B00071_S07665",
    "EN_B00071_S07665",
    "EN_B00062_S04187",
    "EN_B00065_S09873",
    "EN_B00065_S09922",
    "EN_B00084_S02463",
    "EN_B00067_S05066",
    "EN_B00106_S08060",
    "EN_B00073_S06399",
    "EN_B00073_S09236",
    "EN_B00087_S00432",
    "EN_B00085_S05618",
    "EN_B00064_S01262",
    "EN_B00072_S01739",
    "EN_B00059_S03913",
    "EN_B00069_S04036",
    "EN_B00067_S05623",
    "EN_B00060_S05389",
    "EN_B00060_S07290",
    "EN_B00062_S08995",
}
emilia_en_filters = ["ا", "い", "て"]


def deal_with_emilia_audio_dir(audio_dir):
    sub_result, durations = [], []
    vocab_set = set()
    bad_case_de, bad_case_en, bad_case_duration = 0, 0, 0

    glob_filter = os.path.join(audio_dir, "*.json")
    for json_file_path in glob.glob(glob_filter):
        with open(json_file_path, "r") as json_file:
            obj = json.load(json_file)
        text = obj["text"]
        duration = obj["duration"]

        if duration < 0.3 or duration > 30:
            bad_case_duration += 1
            continue
        if obj["language"] == "de":
            if (
                obj["wav"].split("/")[1] in emilia_out_de
                or any(f in text for f in emilia_de_filters)
                or repetition_found(text, length=4)
            ):
                bad_case_de += 1
                continue
        if obj["language"] == "en":
            if (
                obj["wav"].split("/")[1] in emilia_out_en
                or any(f in text for f in emilia_en_filters)
                or repetition_found(text, length=4)
            ):
                bad_case_en += 1
                continue
        if tokenizer == "pinyin":
            text = convert_char_to_pinyin([text], polyphone=polyphone)[0]

        audio_path = str(json_file_path).replace('.json', '.mp3')
        sub_result.append({"audio_path": audio_path, "text": text, "duration": duration})
        durations.append(duration)
        vocab_set.update(list(text))
    return sub_result, durations, vocab_set, bad_case_de, bad_case_en, bad_case_duration


def deal_with_cv_obj(obj, duration):
    text = obj["text"]

    if duration < 0.3 or duration > 30:
        return [], [], set(), 0, 0, 1
    if obj["language"] == "de":
        if (
            obj["wav"] in cv_out_de
            or any(f in text for f in cv_de_filters)
            or repetition_found(text, length=4)
        ):
            # sub_result, durations, vocab_set, bad_case_de, bad_case_en
            return [], [], set(), 0, 1, 0
    if obj["language"] == "en":
        if (
            obj["wav"] in cv_out_en
            or any(f in text for f in cv_en_filters)
            or repetition_found(text, length=4)
        ):
            # sub_result, durations, vocab_set, bad_case_de, bad_case_en
            return [], [], set(), 1, 0, 0
    if tokenizer == "pinyin":
        text = convert_char_to_pinyin([text], polyphone=polyphone)[0]

    audio_path = os.path.join(cv_dataset_dir, obj["language"], "clips", obj["wav"])
    assert os.path.exists(audio_path), f"Audio file does not exist {audio_path}"

    sub_result = [{"audio_path": audio_path, "text": text, "duration": duration}]
    durations = [duration]
    vocab_set = set(list(text))

    return sub_result, durations, vocab_set, 0, 0, 0


def main():
    assert tokenizer in ["pinyin", "char"]
    result = []
    duration_list = []
    text_vocab_set = set()
    total_bad_case_de = 0
    total_bad_case_en = 0
    total_bad_case_duration = 0

    # process raw data
    executor = ProcessPoolExecutor(max_workers=max_workers)
    futures = []
    for lang in emilia_langs:
        emilia_dataset_path = Path(os.path.join(emilia_dataset_dir, lang))
        [
            futures.append(executor.submit(deal_with_emilia_audio_dir, audio_dir))
            for audio_dir in emilia_dataset_path.iterdir()
            if audio_dir.is_dir()
        ]
    
    for futures in tqdm(futures, total=len(futures)):
        sub_result, durations, vocab_set, bad_case_de, bad_case_en, bad_case_duration = futures.result()
        result.extend(sub_result)
        duration_list.extend(durations)
        text_vocab_set.update(vocab_set)
        total_bad_case_de += bad_case_de
        total_bad_case_en += bad_case_en
        total_bad_case_duration += bad_case_duration

    executor.shutdown()

    # Split dataset
    train_ratio, val_ratio, test_ratio = 0.98, 0.01, 0.01
    train_data, temp_data = train_test_split(result, test_size=(1 - train_ratio), random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)

    emilia_splits = {
        "train": train_data,
        "validation": val_data,
        "test": test_data,
    }

    # Process Common Voice dataset
    for lang in cv_langs:
        cv_dataset_path = Path(os.path.join(cv_dataset_dir, lang))

        result = {"train": [], "validation": [], "test": []}

        # Load the clip_durations.tsv file and create a dictionary of durations
        clip_durations_path = cv_dataset_path / "clip_durations.tsv"
        duration_df = pd.read_csv(clip_durations_path, sep="\t")
        duration_map = dict(zip(duration_df["clip"], duration_df["duration[ms]"] / 1000))  # Convert ms to seconds

        for split_name, tsv_file in [("train", "train.tsv"), ("validation", "dev.tsv"), ("test", "test.tsv")]:
            tsv_path = cv_dataset_path / tsv_file
            if os.path.exists(tsv_path):
                df = pd.read_csv(tsv_path, sep="\t")
                for _, row in tqdm(df.iterrows(), total=len(df)):
                    obj = {
                        "wav": row["path"],
                        "text": row["sentence"],
                        "language": row["locale"],
                    }
                    duration = duration_map.get(row["path"], None)
                    if duration is None:
                        raise ValueError(f"Duration is None for path: {['path']}")
                    # check if obj text is nan
                    if obj["text"] != obj["text"]:
                        print(f"Text is nan: {obj['text']}")
                        print(f"Path: {obj['wav']}")
                        print(f"Duration: {duration}")
                        print(f"Language: {obj['language']}")
                        print()
                        print(split_name)
                        continue
                    sub_result, durations, vocab_set, bad_case_de, bad_case_en, bad_case_duration = deal_with_cv_obj(obj, duration)
                    result[split_name].extend(sub_result)
                    duration_list.extend(durations)
                    text_vocab_set.update(vocab_set)
                    total_bad_case_de += bad_case_de
                    total_bad_case_en += bad_case_en
                    total_bad_case_duration += bad_case_duration


    cv_splits = result

    combined_splits = {key: cv_splits[key] + emilia_splits[key] for key in ["train", "validation", "test"]}

    split_durations = {
        "train": [line["duration"] for line in combined_splits["train"]],
        "validation": [line["duration"] for line in combined_splits["validation"]],
        "test": [line["duration"] for line in combined_splits["test"]],
    }

    # save preprocessed dataset to disk
    if not os.path.exists(f"{save_dir}"):
        os.makedirs(f"{save_dir}")
    print(f"\nSaving to {save_dir} ...")

    for split_name, split_data in combined_splits.items():
        os.makedirs(save_dir, exist_ok=True)
        arrow_path = os.path.join(save_dir, f"{split_name}.arrow")
        print(f"Saving {split_name} data to {arrow_path} ...")

        with ArrowWriter(path=arrow_path) as writer:
            for line in tqdm(split_data, desc=f"Writing {split_name} to Arrow"):
                writer.write(line)

    # dump jsons separately saving duration in case for DynamicBatchSampler ease
    for split_name, durations in split_durations.items():
        duration_file = f"{save_dir}/{split_name}_duration.json"
        with open(duration_file, "w", encoding="utf-8") as f:
            json.dump({"duration": durations}, f, ensure_ascii=False)

    # vocab map, i.e. tokenizer
    # add alphabets and symbols (optional, if plan to ft on de/fr etc.)
    # if tokenizer == "pinyin":
    #     text_vocab_set.update([chr(i) for i in range(32, 127)] + [chr(i) for i in range(192, 256)])
    with open(f"{save_dir}/vocab.txt", "w") as f:
        for vocab in sorted(text_vocab_set):
            f.write(vocab + "\n")

    print(f"\nFor {dataset_name}, sample count: {len(result)}")
    print(f"For {dataset_name}, vocab size is: {len(text_vocab_set)}")
    print(f"For {dataset_name}, total {sum(duration_list)/3600:.2f} hours")
    if "DE" in emilia_langs:
        print(f"Bad de transcription case: {total_bad_case_de}")
    if "EN" in emilia_langs:
        print(f"Bad en transcription case: {total_bad_case_en}\n")
    print(f"Bad duration case: {total_bad_case_duration}\n")


def check_vocab():
    original_dataset_name = "Emilia_ZH_EN_pinyin"
    original_save_dir = str(files("f5_tts").joinpath("../../")) + f"/data/{original_dataset_name}"

    original_vocab_path = os.path.join(original_save_dir, "vocab.txt")
    vocab_path = os.path.join(save_dir, "vocab.txt")

    original_vocab = set()
    with open(original_vocab_path) as original_vocab_file:
        for line in original_vocab_file:
            original_vocab.add(line.strip())

    vocab = set()
    with open(vocab_path) as vocab_file:
        for line in vocab_file:
            vocab.add(line.strip())

    print("New vocab is subset:", vocab.issubset(original_vocab))
    print("Missing vocab:", list(vocab.difference(original_vocab)))

    shutil.copyfile(original_vocab_path, vocab_path)


if __name__ == "__main__":
    max_workers = 32

    tokenizer = "pinyin"  # "pinyin" | "char"
    polyphone = True

    cv_langs = ["de"]
    cv_dataset_dir = "/raid/shared/datasets/cv-corpus-19.0-2024-09-13"

    emilia_langs = ["DE"]
    emilia_dataset_dir = "/raid/shared/datasets/Emilia-Dataset"

    dataset_name = f"CV_{'_'.join(cv_langs)}_Emilia_{'_'.join(emilia_langs)}_{tokenizer}"
    save_dir = str(files("f5_tts").joinpath("../../")) + f"/data/{dataset_name}"

    print(f"\nPrepare for {dataset_name}, will save to {save_dir}\n")

    main()
    check_vocab()

    # Emilia               DE
    # samples count       37837916   (after removal)
    # pinyin vocab size       2543   (polyphone)
    # total duration      95281.87   (hours)
    # bad de asr cnt        230435   (samples)
    # bad eh asr cnt         37217   (samples)

    # vocab size may be slightly different due to jieba tokenizer and pypinyin (e.g. way of polyphoneme)
    # please be careful if using pretrained model, make sure the vocab.txt is same
