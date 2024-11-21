import numpy as np


def classify_examples(confidence, aleatoric, dataiq_xthresh, dataiq_ythresh):
    """Classify examples into Easy, Hard, and Ambiguous groups using DataIQ criteria"""
    conf_thresh_low = dataiq_ythresh
    conf_thresh_high = 1 - dataiq_ythresh

    groups = np.empty(len(confidence), dtype=object)

    groups[(confidence >= conf_thresh_high) & (aleatoric <= dataiq_xthresh)] = "Easy"
    groups[(confidence <= conf_thresh_low) & (aleatoric <= dataiq_xthresh)] = "Hard"

    groups[
        ~((confidence >= conf_thresh_high) & (aleatoric <= dataiq_xthresh))
        & ~((confidence <= conf_thresh_low) & (aleatoric <= dataiq_xthresh))
    ] = "Ambiguous"

    # percentile-based thresholding
    # alea_perc = np.percentile(aleatoric, dataiq_xthresh)

    # groups[(confidence >= conf_thresh_high) & (aleatoric <= alea_perc)] = "Easy"
    # groups[(confidence <= conf_thresh_low) & (aleatoric <= alea_perc)] = "Hard"

    # groups[
    #     ~((confidence >= conf_thresh_high) & (aleatoric <= alea_perc))
    #     & ~((confidence <= conf_thresh_low) & (aleatoric <= alea_perc))
    # ] = "Ambiguous"

    # print(f"Using thresholds: {dataiq_xthresh}, {alea_perc}, {dataiq_ythresh}")

    return groups


# def classify_examples(avg_confidence, avg_aleatoric, conf_upper=0.75, conf_lower=0.25, aleatoric_percentile=50):
#     """Classify examples into Easy, Hard, and Ambiguous groups using DataIQ criteria"""
#     alea_perc = np.percentile(avg_aleatoric, aleatoric_percentile)

#     groups = np.empty(len(avg_confidence), dtype=object)
#     groups[(avg_confidence >= conf_upper) & (avg_aleatoric < alea_perc)] = "Easy"
#     groups[(avg_confidence <= conf_lower) & (avg_aleatoric < alea_perc)] = "Hard"
#     groups[
#         ~((avg_confidence >= conf_upper) & (avg_aleatoric < alea_perc))
#         & ~((avg_confidence <= conf_lower) & (avg_aleatoric < alea_perc))
#     ] = "Ambiguous"

#     return groups
