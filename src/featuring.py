import pandas as pd
import numpy as np

from tqdm.auto import tqdm

from sklearn.decomposition import IncrementalPCA


def feature_pca(df, n_components=10):
    pca = IncrementalPCA(n_components=n_components)
    emb = pca.fit_transform(df.values)

    emb_df = pd.DataFrame()

    for i in range(emb.shape[1]):
        emb_df[f"pca_{i}"] = emb[:, i]
    return emb_df 


def calc_per_era(df):
    features = [col for col in df.columns if "feature" in col]
    operators = ["mean", "max", "min", "std"]
    # TODO: concat でつなげる
    feature_dfs = []
    for col in tqdm(features):
        tmp = pd.DataFrame(index=df.index)
        for operator in operators:
            if operator in ["mean", "std"]:
                data_type = np.float16
            else:
                data_type = np.int8
            tmp[f"per_era_{operator}_{col}"] = df.groupby("era")[col].transform(operator).astype(data_type)
        feature_dfs.append(tmp)
    
    df = pd.concat([df] + feature_dfs, axis=1)
    return df

def add_feature(df):
    df = calc_per_era(df)
    features = [col for col in df.columns if "feature" in col]
    df["per_era_count"] = df.groupby("era")[features[0]].transform("count")

    # # exp003
    # # kmeans で特徴量をグルーピングし、グループ間の差と積をとる
    # feature_group = [
    #     ['feature_antistrophic_striate_conscriptionist', 'feature_crosscut_whilom_ataxy', 'feature_departmental_inimitable_sentencer', 'feature_elusive_vapoury_accomplice', 'feature_geminate_crummiest_scourer', 'feature_glandered_unimproved_peafowl', 'feature_hempen_unionist_cone', 'feature_jacobinical_symmetric_roll', 'feature_jewish_stained_disembowelment', 'feature_lacklustre_centroidal_schweitzer', 'feature_limiest_heliolithic_york', 'feature_mendelian_undiscording_avion', 'feature_musicianly_aspirate_creativity', 'feature_reclaimed_insurrectional_moneyer', 'feature_simulated_nonclassified_intercessor', 'feature_snakiest_somalian_wavelet', 'feature_splanchnic_notional_pint', 'feature_stretchy_spiniest_fizgig', 'feature_toltec_korean_disfavourer', 'feature_transisthmian_yogic_linden', 'feature_tridactyl_immoral_snorting', 'feature_trimeter_soggy_greatest', 'feature_unanalyzable_excusable_whirlwind', 'feature_unbreakable_constraining_hegelianism', 'feature_unformed_bent_smatch', 'feature_unsystematized_subcardinal_malaysia', 'feature_willful_sere_chronobiology', 'feature_zoological_peristomial_scute'], 
    #     ['feature_illuminated_gambrel_noria', 'feature_petty_upraised_caddice'],
    #     ['feature_bridal_fingered_pensioner', 'feature_concurring_fabled_adapter', 'feature_dialectal_homely_cambodia', 'feature_donnard_groutier_twinkle', 'feature_maledictive_latter_psellism', 'feature_saddening_unsound_rustling', 'feature_strained_equivocal_phoneme', 'feature_unministerial_unextenuated_teleostean'],
    #     ['feature_bicameral_showery_wallaba'],
    #     ['feature_collectivist_flaxen_gueux', 'feature_pottier_unmanly_collyrium', 'feature_unmodish_zymogenic_rousing'],
    # ]

    # for i in range(len(feature_group)):
    #     for j in range(len(feature_group)):
    #         if i >= j:
    #             continue
    #         
    #         df[f"diff_feature_group_{i}_{j}"] = df[feature_group[i][0]] - df[feature_group[j][0]]
    #         df[f"multiply_feature_group_{i}_{j}"] = df[feature_group[i][0]] * df[feature_group[j][0]]
    return df
