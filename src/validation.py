import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numerai_tools.scoring import correlation_contribution, numerai_corr

from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples


class TimeSeriesSplitGroups(_BaseKFold):
    def __init__(self, n_splits=5):
        super().__init__(n_splits, shuffle=False, random_state=None)

    def split(self, X, y=None, groups=None):
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        group_list = np.unique(groups)
        n_groups = len(group_list)
        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds ={0} greater"
                 " than the number of samples: {1}.").format(n_folds,
                                                            n_groups))
        indices = np.arange(n_samples)
        test_size = (n_groups // n_folds)
        test_starts = range(test_size + n_groups % n_folds,
                            n_groups, test_size)
        test_starts = list(test_starts)[::-1]
        for test_start in test_starts:
            yield (indices[groups.isin(group_list[:test_start])],
                   indices[groups.isin(group_list[test_start:test_start + test_size])])


def caluculate_metrics(dataset_path, valid, benchmark_target="v5_lgbm_cyrusd20"):
    """
    - benchmark_models, meta_model を読み込む
    - join する
    - 可視化する
    - 各種評価指標を計算
    """
    # valid がera, pred, target を持ってるか
    need_cols = ["era", "pred", "target"]
    for col in need_cols:
        if col not in valid.columns:
            raise ValueError(f"Valid dataframe has not {col} columns.")

    benchmark_models_path = os.path.join(dataset_path, "validation_benchmark_models.parquet")
    benchmark = pd.read_parquet(benchmark_models_path, columns=["era", benchmark_target])
    
    meta_model_path = os.path.join(dataset_path, "meta_model.parquet")
    meta_model = pd.read_parquet(meta_model_path)
    valid = valid.reset_index().merge(
        meta_model.reset_index(),
        how="left",
        on=["id", "era"]
    ).merge(
        benchmark.reset_index(),
        how="left",
        on=["id", "era"]
    ).set_index("id")

    per_era_corr = valid[["era", "pred", "target"]].dropna().groupby("era").apply(
        lambda x: numerai_corr(x[["pred"]].dropna(), x["target"].dropna())
    )

    per_era_bmc = valid[["era", "target", "pred", benchmark_target]].dropna().groupby("era").apply(
        lambda x: correlation_contribution(x[["pred"]], x[benchmark_target], x["target"])
    )


    per_era_mmc = valid[["era", "target", "pred", "numerai_meta_model"]].dropna().groupby("era").apply(
        lambda x: correlation_contribution(x[["pred"]], x["numerai_meta_model"], x["target"])
    )

    _, ax = plt.subplots(3, 1, figsize=(10, 15))
    per_era_corr.cumsum().plot(
        title="Validation CORR",
        kind="bar",
        xticks=[],
        legend=False,
        snap=False,
        ax=ax[0]
    )

    per_era_bmc.cumsum().plot(
        title="Validation BMC",
        kind="bar",
        xticks=[],
        legend=False,
        snap=False,
        ax=ax[1]
    )

    per_era_mmc.cumsum().plot(
        title="Validation MMC",
        kind="bar",
        xticks=[],
        legend=False,
        snap=False,
        ax=ax[2]
    )

    plt.show()

    corr_mean = per_era_corr.mean()
    corr_std = per_era_corr.std()
    corr_sharpe = corr_mean / corr_std
    corr_max_drawdown = (per_era_corr.cumsum().expanding(min_periods=1).max() - per_era_corr.cumsum()).max()
     
    mmc_mean = per_era_mmc.mean()
    mmc_std = per_era_mmc.std()
    mmc_sharpe = mmc_mean / mmc_std
    mmc_max_drawdown = (per_era_mmc.cumsum().expanding(min_periods=1).max() - per_era_mmc.cumsum()).max()

    bmc_mean = per_era_bmc.mean()
    bmc_std = per_era_bmc.std()
    bmc_sharpe = bmc_mean / bmc_std
    bmc_max_drawdown = (per_era_bmc.cumsum().expanding(min_periods=1).max() - per_era_bmc.cumsum()).max()

    return (
        pd.DataFrame({
            "corr_mean": [corr_mean.values[0]],
            "corr_std": [corr_std.values[0]],
            "corr_shape": [corr_sharpe.values[0]],
            "corr_max_drawdown": [corr_max_drawdown.values[0]],
            
            "mmc_mean": [mmc_mean.values[0]],
            "mmc_std": [mmc_std.values[0]],
            "mmc_shape": [mmc_sharpe.values[0]],
            "mmc_max_drawdown": [mmc_max_drawdown.values[0]],

            "bmc_mean": [bmc_mean.values[0]],
            "bmc_std": [bmc_std.values[0]],
            "bmc_shape": [bmc_sharpe.values[0]],
            "bmc_max_drawdown": [bmc_max_drawdown.values[0]],
        }),
        per_era_corr,
        per_era_bmc,
        per_era_mmc
    )
        


