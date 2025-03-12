# 目標
- numerai データセットの特徴量を bot に使用する
- numerai データセットの特徴量生成を bot に活かす
- ロバストなモデリング、評価方法を numerai から学ぶ

# numerai tournament の特徴
- 特徴量、目的変数は 5分位化されている
- 特徴量同士の相関は除去されている
- era は 1週間単位
- 特徴量は秘匿化されているが、頑張れば何の数値か推定することが可能

# 参考資料
- https://qiita.com/blog_UKI/items/fb401725288e58c92bd6
    - 機械学習による株価予測 はじめようNumerai
    - UKI さんの Numerai 実践記も含まれる
    - 指標や特徴量に関する考察が豊富
- https://zenn.dev/motion/scraps/604c6bf13d59a7
    - 仮想通貨MLbot探求のしおり
    - 特徴量生成、目的変数など広範にわたる知見が紹介されている
- https://colab.research.google.com/github/numerai/example-scripts/blob/master/feature_neutralization.ipynb
    - numerai tutorial の feature neutralization に関する考察
    - やりすぎると予測精度が落ちる、
    
# メモ
## model_id の一覧を取得

```
# model の UUID を取得
models = napi.get_models()
```

- key: model_name, value: uuid
- submission 提出時の model_id は uuid を使う


# TODO
## submission

- 後処理などが異なるケースに対応させる
    - 後処理関数を作って使用有無を実行引数に入れる
- クラウド実行させる
    - Cloud Run
    - デプロイを簡単にする
- config ファイルを ディレクトリに押し込んで、for loop で全部回す

## 学習環境
- 学習パイプラインを作る
    - 学習曲線を可視化
    - corr などの評価値を計算
- `training_config.json` に特徴量のリストも加える
- `submissions/main.py` で `training_config.json` の特徴量リストを呼び出して使う
- `submissions/main.py` で アンサンブルに対応する（kfold でモデルが複数あるケース）

# Done
- config が異なるケースに対応させる
    - 実行引数にパスを渡す
- check_round_open でラウンド開始してるかチェック
    - False のときは1分くらい待機する