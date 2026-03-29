import pandas as pd
import numpy as np
from scipy.stats import trim_mean
from copy import deepcopy
from gcrl_landscapes.util.eval import cvar, iqr


DIM_LABEL_MAPPING = {
    "actor_p_trajgoal": "$p_{trajgoal}$",
    "discount": "Discount Factor",
    "alpha": "Alpha",
    "lr": "Learning Rate",
    "lr-uniform": "Learning Rate",
}

FTU_SIGNIFICANCE_THRESHOLD = 0.05

CVAR_CONFIDENCE_LEVELS = (10, 20, 30, 40)
DISP_CONFIDENCE_LEVELS = (10, 90)


def map_labels(label: str) -> str:
    return DIM_LABEL_MAPPING[label] if label in DIM_LABEL_MAPPING else label


def resolve_alpha_sync(df: pd.DataFrame, hyperparameters: list[str]) -> pd.DataFrame:
    """Drop hp.low_alpha and hp.high_alpha columns when they were synced from alpha.

    When only 'alpha' is in hyperparameters (not 'low_alpha'/'high_alpha'), the
    configurations module syncs low_alpha/high_alpha to the same value as alpha.
    These duplicated columns would be misidentified as independently varied HPs.
    """
    if (
        "alpha" in hyperparameters
        and "low_alpha" not in hyperparameters
        and "high_alpha" not in hyperparameters
    ):
        cols_to_drop = [c for c in ["hp.low_alpha", "hp.high_alpha"] if c in df.columns]
        return df.drop(columns=cols_to_drop)
    return df


def compute_additional_information(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Compute additional metrics which can be calculated given the data. E.g. CVaR, Dispersion

    Args:
        phase_results: all phase results in list

    Returns:
        new object with additional information with same layout as original one
    """
    results_copy = deepcopy(results)
    # enumerate phases
    results_copy["phase_num"] = results_copy["phase"].rank(method="dense").astype(int)
    # Normalize goal end distances by start distance to get a distance in [1, inf)
    results_copy["normalized_goal_distances"] = results_copy["eval_result"].apply(
        lambda results_per_seed: np.array(
            [
                np.array(result["goal_end_distances"])
                / np.array(result["goal_start_distances"])
                for result in results_per_seed.info
            ]
        ).reshape(-1)
    )
    # To see this as a "score", subtract from 1. Yields scores in (-inf, 1], where 1 is perfect, 0 neutral and below is bad
    results_copy["normalized_goal_distance_returns"] = results_copy[
        "normalized_goal_distances"
    ].apply(lambda distances: 1 - distances)
    results_copy["mean_normalized_goal_distance"] = results_copy[
        "normalized_goal_distances"
    ].apply(np.mean)
    results_copy["mean_normalized_goal_distance_return"] = results_copy[
        "normalized_goal_distance_returns"
    ].apply(np.mean)

    # Calculate CVaR
    for confidence_level in CVAR_CONFIDENCE_LEVELS:
        results_copy[f"cvar{confidence_level}_normalized_goal_distance_return"] = (
            results_copy["normalized_goal_distance_returns"].apply(
                lambda distances: cvar(distances, confidence_level=confidence_level)
            )
        )

    # Calculate Dispersion on goal distance distribution
    results_copy["disp_normalized_goal_distance"] = results_copy[
        "normalized_goal_distance_returns"
    ].apply(lambda returns: iqr(returns, DISP_CONFIDENCE_LEVELS))
    results_copy["disp_normalized_goal_distance_score"] = (
        1 - results_copy["disp_normalized_goal_distance"]
    )
    assert results_copy["disp_normalized_goal_distance_score"].max() <= 1

    # Regret
    ## Check how many hyperparameters are varied.
    ## This only supports 2 HPs at the same time.
    ## Otherwise we'll have to also look at the different combinations and apply some kind of aggregation
    temp_df = results_copy.loc[:, results_copy.columns.str.startswith("hp.")]
    if sum(temp_df.nunique() > 1) > 2:
        raise NotImplementedError(
            "More than 2 hyperparameters are varied. This is not supported yet for regret calculations as these are done upfront."
        )
    ## Now calculate regret
    for column in (
        "mean_normalized_goal_distance_return",
        "success",
    ):
        column_max_per_phase = (
            (  # marginalize seed
                results_copy.groupby(by=["eval_step", "phase", "config_index"])[column]
                .apply(
                    lambda column_values: trim_mean(column_values, proportiontocut=0.25)
                )
                .rename(f"{column}_iqm")  # type: ignore
                .reset_index()
            )
            .groupby(by=["eval_step", "phase"])[f"{column}_iqm"]
            .max()
            .reset_index()
            .rename(columns={f"{column}_iqm": f"{column}_iqm_max_per_phase"})
        )  # get best configuration per phase
        results_copy = pd.merge(
            results_copy,
            column_max_per_phase,
            how="left",
            on=["eval_step", "phase"],
        )
        results_copy[f"{column}_regret"] = (
            results_copy[f"{column}_iqm_max_per_phase"] - results_copy[column]
        )
        results_copy[f"{column}_normalized_regret"] = (
            results_copy[f"{column}_regret"]
            / results_copy[f"{column}_iqm_max_per_phase"]
        )
        ## Calculate cumulative regret
        cum_regret_result: pd.DataFrame = results_copy[  # type: ignore
            (results_copy["eval_step"] == results_copy["phase"])
        ].sort_values(by="phase")
        col_cummean = (
            cum_regret_result.groupby(by=["config_index", "seed"])[f"{column}_regret"]
            .expanding()
            .mean()
            .reset_index(level=[0, 1], drop=True)
            .rename(f"{column}_regret_cummean")  # type: ignore
        )  # type: ignore
        col_normalized_cummean = (
            cum_regret_result.groupby(by=["config_index", "seed"])[
                f"{column}_normalized_regret"
            ]
            .expanding()
            .mean()
            .reset_index(level=[0, 1], drop=True)
            .rename(f"{column}_normalized_regret_cummean")  # type: ignore
        )
        results_copy = pd.merge(
            results_copy,
            col_cummean,
            how="outer",
            left_index=True,
            right_index=True,
        )
        results_copy = pd.merge(
            results_copy,
            col_normalized_cummean,
            left_index=True,
            right_index=True,
        )

    return results_copy


def merge_experiments(
    results: dict[str, tuple[dict, pd.DataFrame, pd.DataFrame]],
    # results: dict[str, tuple[dict, ResultsPerStep[PhaseResult]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.concat(
        [
            result.assign(
                agent=run_info["arguments"]["agent"],
                dataset=",".join(run_info["arguments"]["datasets"])
                if "datasets" in run_info["arguments"]
                else run_info["arguments"]["dataset"],
                constant_dataset=len(set(run_info["arguments"]["datasets"])) == 1,
                hps=lambda x: [frozenset(run_info["arguments"]["hyperparameters"])]
                * len(x),
            )
            for run_info, result, train_log in results.values()
        ]
    ), pd.concat(
        [
            train_log.assign(
                agent=run_info["arguments"]["agent"],
                dataset=",".join(run_info["arguments"]["datasets"])
                if "datasets" in run_info["arguments"]
                else run_info["arguments"]["dataset"],
                constant_dataset=len(set(run_info["arguments"]["datasets"])) == 1,
                hps=lambda x: [frozenset(run_info["arguments"]["hyperparameters"])]
                * len(x),
            )
            for run_info, result, train_log in results.values()
        ]
    )


def calculate_regret_for_experiment(
    df: pd.DataFrame, regret_column: str, base_column: str
) -> pd.DataFrame:
    """Calculate the different regret metrics for one experiment.
    Calculates metrics for best configuration per phase.

    Args:
        df: pandas dataframe of a single experiment

    Returns:
        A dataframe with regret metrics. It will have a row for each phase and columns for the regret in future phases. The row will only reflect the best configuration for that phase. Additionally there is a column with the cumulation of future regrets.
    """
    # Marginalize Seed
    df = (
        df.groupby(by=["phase_num", "config_index"])
        .agg(
            {  # type: ignore
                column: lambda column_values: trim_mean(
                    column_values, proportiontocut=0.25
                )
                for column in [regret_column, base_column]
            }
        )
        .reset_index()
    )

    regret_df = pd.DataFrame(
        columns=[f"regret_phase_{i + 1}" for i in range(len(df["phase_num"].unique()))]
        + ["mean_regret_over_phases", "mean_future_regret_over_phases"]
    )
    regret_df.index = regret_df.index.rename("phase")  # type: ignore
    all_phases = np.sort(df["phase_num"].unique())  # type: ignore

    for phase_idx, phase_num in enumerate(df["phase_num"].unique()):
        best_config_index = (
            df[df["phase_num"] == phase_num]  # type: ignore
            .sort_values(base_column, ascending=False)
            .iloc[0]["config_index"]
        )
        best_config_df = df[df["config_index"] == best_config_index]
        assert best_config_df["phase_num"].nunique() == len(all_phases)  # type: ignore
        assert len(best_config_df["phase_num"]) == len(all_phases)
        regret_df.loc[phase_num] = [np.nan] * (len(all_phases) + 2)
        for other_phase_idx, other_phase_num in enumerate(df["phase_num"].unique()):
            regret_df.loc[phase_num, f"regret_phase_{other_phase_num}"] = (
                best_config_df[best_config_df["phase_num"] == other_phase_num].iloc[0][  # type: ignore
                    regret_column
                ]
            )
        regret_df.loc[phase_num, "mean_regret_over_phases"] = best_config_df[
            regret_column
        ].mean()
        regret_df.loc[phase_num, "mean_future_regret_over_phases"] = best_config_df[
            best_config_df["phase_num"] > phase_num
        ][regret_column].mean()

    return regret_df
