import os
import re
import random
import json

import pandas as pd
from torchvision.transforms import v2
import torchvision.transforms.functional as TF

from utils.configs import Configs
from utils.plotting import plot_top_k_predictions, stackRGB
from utils.utils import open_file_as_tensor, open_file_as_tensor_norm


def save_fig_as_img(path, mode, out_name, dpi=300):
    if os.path.exists(out_name):
        return

    tensor = open_file_as_tensor_norm(path, f"self_{mode}_rgb")

    if mode == "PS":
        tensor = v2.CenterCrop(size=(256, 256))(tensor)
        rgb = stackRGB(0, 1, 2, tensor)
    elif mode == "S2":
        tensor = v2.CenterCrop(size=(64, 64))(tensor)
        rgb = stackRGB(1, 2, 3, tensor)

    img = TF.to_pil_image(rgb)
    img.save(out_name, format="PNG")
    print(f"Saved to {out_name}")


config = Configs('config/setup_12.json')

if os.path.exists(config.pred_path):
    df = pd.read_csv(config.pred_path, names=["PS", "S2", "Pred_sim", "Match_label"],
                     dtype={"PS": str, "S2": str, "Pred_sim": float, "Match_label": int})
else:
    raise FileExistsError(f"No such file: {config.pred_path}")

found = 0
questions = []

config.normalise = False

p = re.compile(r"^.*/(.*)\.tif$")
questions = []
num_of_questions = 20

PS_uniques = []

while found < num_of_questions:
    PS_unique = df.sample(1).PS.item()
    if PS_unique in PS_uniques:
        continue
    else:
        PS_uniques.append(PS_unique)
    df_subset = df[df.PS == PS_unique].copy()

    df_subset.sort_values(by="Pred_sim", ascending=False, inplace=True)
    df_subset.reset_index(drop=True, inplace=True)
    sum(df_subset.iloc[0:5, -1]) > 0

    if sum(df_subset.iloc[0:5, -1]) > 0:
        model_correct = sum(df_subset.iloc[0:3, -1]) > 0
        found += 1
        questionImage = {}
        df_subset['topk'] = df_subset.index
        df_subset['topk'] = df_subset['topk'].apply(lambda x: x + 1)

        # Create and save PS image
        PS_path = df_subset.iloc[0, 0]
        PS_img_path = p.sub(r"game/quiz/imgs/\1.png", PS_path)
        questionImage["questionImage"] = {"src": p.sub(r"imgs/\1.png", PS_path), "modelCorrect": model_correct}
        save_fig_as_img(PS_path, "PS", PS_img_path)

        top_pred = df_subset.iloc[0:5, 1:]
        random_df = df_subset[5:150].sample(7)
        df_subset = pd.concat([top_pred, random_df.iloc[:, 1:]], ignore_index=True)
        df_subset = df_subset.rename(columns={"Match_label": "correct", "Pred_sim": "similarity"})
        df_subset['src'] = df_subset['S2'].apply(lambda x: p.sub(r"imgs/\1.png", x))
        df_subset['img_path'] = df_subset['S2'].apply(lambda x: p.sub(r"game/quiz/imgs/\1.png", x))

        df_subset['correct'] = df_subset['correct'].astype(bool)
        df_subset['similarity'] = df_subset['similarity'].round(2).astype(str)
        df_subset['topk'] = df_subset['topk'].astype(str)

        for S2_path, img_path in zip(df_subset.S2, df_subset.img_path):
            save_fig_as_img(S2_path, "S2", img_path)

        images = df_subset.iloc[:, 1:-1].to_dict("records")
        random.shuffle(images)
        questionImage["images"] = images
        questions.append(questionImage)

with open("game/quiz/questions.json", "w") as f:
    json.dump(questions, f)
