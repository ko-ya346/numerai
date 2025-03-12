from glob import glob

import os
import sys
import json
import yaml
import pickle
import pandas as pd
import numpy as np

from numerapi import NumerAPI

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(SCRIPT_DIR)
sys.path.append(os.path.join(SCRIPT_DIR, "../"))
from src.model import LightGBMModel
from src.featuring import add_feature


def load_yaml(config_path="config.yaml"):
    """
    yaml ファイル読み込み
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_json(config_path: str):
    with open(config_path, "r") as f:
        return json.load(f)


def main():
    env = load_yaml(os.path.join(SCRIPT_DIR, "env.yaml"))
    # 現在の round を取得
    napi = NumerAPI(env["api_key"], env["api_secret"])
    
    numerai_models = napi.get_models()

    live_filename = f"v5.0/live.parquet"
    
    current_round = napi.get_current_round() 

    # round 毎のディレクトリを用意
    round_path = os.path.join(SCRIPT_DIR, f"round{current_round}")
    os.makedirs(round_path, exist_ok=True)
    
    # live データをダウンロード
    live_data_path = os.path.join(round_path, "live.parquet")
    napi.download_dataset(live_filename, live_data_path)

    # live データ読み込み
    live = pd.read_parquet(
        live_data_path,
    )
    live = add_feature(live)
    print(live.shape)

    # Load Config
    config_paths = sorted(list(glob(os.path.join(SCRIPT_DIR, "config/*.yaml"))))
    for config_path in config_paths:
        print("=" * 64)
        print("config path: ", config_path)
        # submission の設定ファイルを読み込む
        sub_config = load_yaml(os.path.join(SCRIPT_DIR, config_path))

        # key: model_name, value: model_id (uuid)
        model_id = numerai_models[sub_config["model_name"]]
        
        if isinstance(sub_config["model_path"], str):
            exp_model_paths = [sub_config["model_path"]]
        elif isinstance(sub_config["model_path"], list):
            exp_model_paths = sub_config["model_path"]
        else:
            raise ValueError(f"Unknown model_path: {sub_config['model_path']}")
        print(exp_model_paths)

        pred = np.zeros(len(live))

        # 複数の実験でアンサンブル出来るようにする
        for exp_model_path in exp_model_paths:
            print("now: ", exp_model_path)
            # 特徴量を読み込む
            with open(os.path.join(SCRIPT_DIR, exp_model_path, "feature.pkl"), "rb") as f:
                features = pickle.load(f)

            # モデル読み込み
            model_paths = list(glob(os.path.join(SCRIPT_DIR, exp_model_path, "model*.pkl")))
            for model_path in model_paths:
                model = LightGBMModel()
                model.load_model(model_path)
                print(model.model)
                pred += model.predict(live[features]) / len(model_paths) / len(model_paths)

        # 提出
        submission = pd.Series(
            pred,
            index=live.index
        ).to_frame("prediction")

        submission_path = os.path.join(round_path, f"prediction-{sub_config['model_name']}-{current_round}.csv")
        submission.to_csv(submission_path)
        napi.upload_predictions(submission_path, model_id=model_id)

    return

if __name__ == "__main__":
    main()
