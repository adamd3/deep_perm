import numpy as np


def classify_examples(confidence, aleatoric, dips_xthresh, dips_ythresh):
    """Classify examples into Easy, Hard, and Ambiguous groups using DataIQ criteria"""
    conf_thresh_low = dips_ythresh
    conf_thresh_high = 1 - dips_ythresh

    alea_perc = np.percentile(aleatoric, dips_xthresh)

    groups = np.empty(len(confidence), dtype=object)

    # groups[(confidence >= conf_thresh_high) & (aleatoric <= dips_xthresh)] = "Easy"
    # groups[(confidence <= conf_thresh_low) & (aleatoric <= dips_xthresh)] = "Hard"

    # groups[
    #     ~((confidence >= conf_thresh_high) & (aleatoric <= dips_xthresh))
    #     & ~((confidence <= conf_thresh_low) & (aleatoric <= dips_xthresh))
    # ] = "Ambiguous"

    groups[(confidence >= conf_thresh_high) & (aleatoric <= alea_perc)] = "Easy"
    groups[(confidence <= conf_thresh_low) & (aleatoric <= alea_perc)] = "Hard"

    groups[
        ~((confidence >= conf_thresh_high) & (aleatoric <= alea_perc))
        & ~((confidence <= conf_thresh_low) & (aleatoric <= alea_perc))
    ] = "Ambiguous"

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
