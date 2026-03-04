"""
analysis_utils.py

Reusable statistical utilities for vaccination analysis notebooks.

Includes:
- Data subsetting and cleaning
- Vaccination recoding
- Weighted chi-square + Cramer's V
- Weighted logistic regression (survey weights via statsmodels GLM)
- Weighted ordinal logistic regression (survey weights via R svyolr)
- Odds ratio extraction
- Reusable visualization helpers
"""

# ============================================================
# Imports
# ============================================================

import pandas as pd
import numpy as np

from typing import List, Tuple, Optional

from scipy.stats import chi2_contingency

import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel


# ============================================================
# Data Utilities
# ============================================================

def subset_variables(
    df: pd.DataFrame,
    columns: List[str],
    weight_col: Optional[str] = None
) -> pd.DataFrame:
    """Return copy of selected columns (optionally including weight column)."""
    cols = columns.copy()
    if weight_col and weight_col not in cols:
        cols.append(weight_col)
    return df[cols].copy()


def drop_missing(
    df: pd.DataFrame,
    columns: List[str]
) -> pd.DataFrame:
    """Drop rows missing any of the specified columns."""
    return df.dropna(subset=columns)


def recode_vaccination_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary and ordinal vaccination variables.
    Assumes sanitized column names:
        - 'Vaccination_at_Birth'
        - 'Vaccination_Flag'
    """
    df = df.copy()

    if "Vaccination_at_Birth" in df.columns:
        df["Vaccination_at_Birth_bin"] = df["Vaccination_at_Birth"].astype(int)

    if "Vaccination_Flag" in df.columns:
        df["Vaccination_Flag_ord"] = df["Vaccination_Flag"].map({
            "None Post Birth": 0,
            "First": 1,
            "Second": 2,
            "Full": 3
        })

    return df


# ============================================================
# Statistical Tests
# ============================================================

def chi2_cramers_v_weighted(
    df: pd.DataFrame,
    x: str,
    y: str,
    weight_col: str
) -> dict:
    """
    Compute weighted chi-square test and Cramer's V.
    """
    table = pd.pivot_table(
        df,
        index=x,
        columns=y,
        values=weight_col,
        aggfunc="sum",
        fill_value=0
    )

    chi2, p, dof, _ = chi2_contingency(table)

    n = table.to_numpy().sum()
    cramers_v = np.sqrt((chi2 / n) / (min(table.shape) - 1))

    return {
        "chi2": chi2,
        "p_value": p,
        "dof": dof,
        "cramers_v": cramers_v,
        "table": table
    }


def normalized_crosstab(
    df: pd.DataFrame,
    row: str,
    col: str
) -> pd.DataFrame:
    """Return row-normalized contingency table."""
    table = pd.crosstab(df[row], df[col])
    return table.div(table.sum(axis=1), axis=0)


def screen_categorical_associations(
    df: pd.DataFrame,
    outcome: str,
    weight_col: str
) -> pd.DataFrame:
    """
    Run weighted chi-square tests against all categorical predictors.
    """
    results = []

    for col in df.columns:
        if col in [outcome, weight_col]:
            continue

        if df[col].nunique() < 20:
            stats = chi2_cramers_v_weighted(df, col, outcome, weight_col)

            results.append({
                "variable": col,
                "chi2": stats["chi2"],
                "p_value": stats["p_value"],
                "cramers_v": stats["cramers_v"]
            })

    return pd.DataFrame(results).sort_values("p_value")


# ============================================================
# Modeling Utilities
# ============================================================

def fit_weighted_logit(
    df: pd.DataFrame,
    formula: str,
    weight_col: str
):
    """
    Fit survey-weighted binary logistic regression using statsmodels GLM.

    Parameters
    ----------
    df : pd.DataFrame
    formula : str
        Patsy formula string, e.g. 'y ~ x1 + C(x2)'
    weight_col : str
        Column name of survey/sampling weights.

    Returns
    -------
    statsmodels GLMResultsWrapper
    """
    model = sm.GLM.from_formula(
        formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df[weight_col]
    )
    return model.fit()


def fit_weighted_ordinal_logit(
    df: pd.DataFrame,
    outcome: str,
    predictors: List[str],
    weight_col: str
):
    """
    Fit survey-weighted ordinal logistic regression via R's svyolr.

    Requires rpy2 and the R 'survey' package to be installed.

    Parameters
    ----------
    df : pd.DataFrame
    outcome : str
        Column name of the ordinal outcome variable.
    predictors : list of str
        Column names of predictors.
    weight_col : str
        Column name of survey/sampling weights.

    Returns
    -------
    pd.DataFrame
        Odds ratios with 95% CI and p-values.
    """
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter

    survey = importr("survey")
    base = importr("base")

    # Transfer dataframe to R using modern conversion context
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df = ro.conversion.py2rpy(df)

    ro.globalenv["r_df"] = r_df

    # Build formula string
    formula_str = f"{outcome} ~ {' + '.join(predictors)}"
    ro.globalenv["formula_str"] = formula_str
    ro.globalenv["weight_col"] = weight_col

    # Pass outcome column name to R
    ro.globalenv["outcome_col"] = outcome

    # Run svyolr in R
    ro.r("""
        library(survey)
        r_df[[outcome_col]] <- as.factor(r_df[[outcome_col]])
        r_df[[weight_col]] <- as.numeric(r_df[[weight_col]])
        design <- svydesign(ids = ~1, weights = as.formula(paste0("~", weight_col)), data = r_df)
        model <- svyolr(as.formula(formula_str), design = design)
        coefs <- coef(summary(model))
    """)

    # Pull results back to Python
    coefs = ro.globalenv["coefs"]
    with localconverter(ro.default_converter + pandas2ri.converter):
        coefs_df = ro.conversion.rpy2py(base.as_data_frame(coefs))

    coefs_df.columns = ["coef", "Std_Error", "t_value"]
    coefs_df["OR"] = np.exp(coefs_df["coef"])
    coefs_df["CI_lower"] = np.exp(coefs_df["coef"] - 1.96 * coefs_df["Std_Error"])
    coefs_df["CI_upper"] = np.exp(coefs_df["coef"] + 1.96 * coefs_df["Std_Error"])

    return coefs_df[["OR", "CI_lower", "CI_upper", "t_value"]]


# ============================================================
# Model Interpretation
# ============================================================

def extract_odds_ratios(result) -> pd.DataFrame:
    """
    Extract odds ratios and 95% CI from statsmodels result.
    """
    params = result.params
    conf = result.conf_int()

    or_df = pd.DataFrame({
        "OR": np.exp(params),
        "CI_lower": np.exp(conf[0]),
        "CI_upper": np.exp(conf[1]),
        "p_value": result.pvalues
    })

    return or_df


# ============================================================
# Visualization Utilities
# ============================================================

def plot_heatmap(table: pd.DataFrame, title: str = ""):
    """Plot heatmap of contingency table."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8, 6))
    sns.heatmap(table, annot=True, fmt=".2f", cmap="Blues")
    plt.title(title)
    plt.tight_layout()
    return plt.gca()


def plot_odds_ratios(or_df: pd.DataFrame, title: str = ""):
    """Plot odds ratios with confidence intervals."""
    import matplotlib.pyplot as plt

    df = or_df.copy()
    df = df.dropna()

    plt.figure(figsize=(8, 6))
    plt.errorbar(
        df["OR"],
        df.index,
        xerr=[df["OR"] - df["CI_lower"], df["CI_upper"] - df["OR"]],
        fmt="o"
    )
    plt.axvline(1, linestyle="--")
    plt.title(title)
    plt.xlabel("Odds Ratio")
    plt.tight_layout()
    return plt.gca()


def plot_weighted_coverage_bar(
    coverage_df: pd.DataFrame,
    x: str,
    y: str,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "Weighted Probability",
    color: str = "#4c72b0",
    ylim: tuple = (0, 1.25),
    figsize: tuple = (16, 10),
    save_path: Optional[str] = None
):
    """
    Bar plot of weighted coverage with percentage annotations.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    plt.figure(figsize=figsize)

    ax = sns.barplot(data=coverage_df, x=x, y=y, color=color, edgecolor='none')

    ax.set_ylabel(ylabel, fontsize=32, labelpad=20)
    ax.set_xlabel(xlabel or x, fontsize=32, labelpad=20)
    ax.set_ylim(*ylim)
    ax.tick_params(labelsize=26)

    for p in ax.patches:
        h = p.get_height()
        ax.annotate(f'{h*100:.1f}%',
                    (p.get_x() + p.get_width() / 2., h),
                    ha='center', va='center', xytext=(0, 18),
                    textcoords='offset points', fontsize=22)

    if title:
        ax.set_title(title, fontsize=32)

    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    return ax


def plot_stacked_vaccination(
    vax_comp: pd.DataFrame,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "Proportion",
    palette: str = "crest",
    ylim: tuple = (0, 1.3),
    figsize: tuple = (16, 10),
    save_path: Optional[str] = None
):
    """
    Stacked bar plot of vaccination progression proportions.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    main_palette = sns.color_palette(palette, len(vax_comp.columns))

    fig, ax = plt.subplots(figsize=figsize)
    vax_comp.plot(
        kind='bar',
        stacked=True,
        color=main_palette,
        edgecolor='white',
        linewidth=1.5,
        width=0.75,
        ax=ax
    )

    ax.set_ylabel(ylabel, fontsize=32, labelpad=20)
    ax.set_xlabel(xlabel, fontsize=32, labelpad=20)
    ax.set_ylim(*ylim)
    ax.tick_params(labelsize=26, rotation=0)
    ax.legend(title="", fontsize=22, loc="upper center",
              bbox_to_anchor=(0.5, 1.15), ncol=len(vax_comp.columns), frameon=False)

    if title:
        ax.set_title(title, fontsize=32)

    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    return ax