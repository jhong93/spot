import sys
from collections import defaultdict
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt


def parse_ground_truth(truth):
    label_dict = defaultdict(lambda: defaultdict(list))
    for x in truth:
        for e in x['events']:
            label_dict[e['label']][x['video']].append(e['frame'])
    return label_dict


def get_predictions(pred, label=None):
    flat_pred = []
    for x in pred:
        for e in x['events']:
            if label is None or e['label'] == label:
                flat_pred.append((x['video'], e['frame'], e['score']))
    flat_pred.sort(key=lambda x: x[-1], reverse=True)
    return flat_pred


def compute_average_precision(
        pred, truth, tolerance=0, min_precision=0,
        plot_ax=None, plot_label=None, plot_raw_pr=True
):
    total = sum([len(x) for x in truth.values()])
    recalled = set()

    # The full precision curve has TOTAL number of bins, when recall increases
    # by in increments of one
    pc = []
    _prev_score = 1
    for i, (video, frame, score) in enumerate(pred, 1):
        assert score <= _prev_score
        _prev_score = score

        # Find the ground truth frame that is closest to the prediction
        gt_closest = None
        for gt_frame in truth.get(video, []):
            if (video, gt_frame) in recalled:
                continue
            if gt_closest is None or (
                    abs(frame - gt_closest) > abs(frame - gt_frame)
            ):
                gt_closest = gt_frame

        # Record precision each time a true positive is encountered
        if gt_closest is not None and abs(frame - gt_closest) <= tolerance:
            recalled.add((video, gt_closest))
            p = len(recalled) / i
            pc.append(p)

            # Stop evaluation early if the precision is too low.
            # Not used, however when nin_precision is 0.
            if p < min_precision:
                break

    interp_pc = []
    max_p = 0
    for p in pc[::-1]:
        max_p = max(p, max_p)
        interp_pc.append(max_p)
    interp_pc.reverse()     # Not actually necessary for integration

    if plot_ax is not None:
        rc = np.arange(1, len(pc) + 1) / total
        if plot_raw_pr:
            plot_ax.plot(rc, pc, label=plot_label, alpha=0.8)
        plot_ax.plot(rc, interp_pc, label=plot_label, alpha=0.8)

    # Compute AUC by integrating up to TOTAL bins
    return sum(interp_pc) / total


def compute_mAPs(
        truth, pred, tolerances=[0, 1, 2, 4], plot_pr=False
):
    assert {v['video'] for v in truth} == {v['video'] for v in pred}, \
        'Video set mismatch!'

    truth_by_label = parse_ground_truth(truth)

    fig, axes = None, None
    if plot_pr:
        fig, axes = plt.subplots(
            len(truth_by_label), len(tolerances), sharex=True, sharey=True,
            figsize=(16, 16))

    class_aps_for_tol = []
    mAPs = []
    for i, tol in enumerate(tolerances):
        class_aps = []
        for j, (label, truth_for_label) in enumerate(
            sorted(truth_by_label.items())
        ):
            ap = compute_average_precision(
                get_predictions(pred, label=label),
                truth_for_label, tolerance=tol,
                plot_ax=axes[j, i] if axes is not None else None)
            class_aps.append((label, ap))
        mAP = np.mean([x[1] for x in class_aps])
        mAPs.append(mAP)
        class_aps.append(('mAP', mAP))
        class_aps_for_tol.append(class_aps)

    header = ['AP @ tol'] + tolerances
    rows = []
    for c, _ in class_aps_for_tol[0]:
        row = [c]
        for class_aps in class_aps_for_tol:
            for c2, val in class_aps:
                if c2 == c:
                    row.append(val * 100)
        rows.append(row)
    print(tabulate(rows, headers=header, floatfmt='0.2f'))

    print('Avg mAP (across tolerances): {:0.2f}'.format(np.mean(mAPs) * 100))

    if plot_pr:
        for i, tol in enumerate(tolerances):
            for j, label in enumerate(sorted(truth_by_label.keys())):
                ax = axes[j, i]
                ax.set_xlabel('Recall')
                ax.set_xlim(0, 1)
                ax.set_ylabel('Precision')
                ax.set_ylim(0, 1.01)
                ax.set_title('{} @ tol={}'.format(label, tol))
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    sys.stdout.flush()
    return mAPs, tolerances