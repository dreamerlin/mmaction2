from itertools import groupby

import numpy as np

from mmaction.core import temporal_iou


def load_localize_proposal_file(filename):
    """Load the proposal file and split it into many parts which contain one
    video's information separately.

    Args:
        filename(str): Path to the proposal file.

    Returns:
        list: List of all videos' information.
    """
    lines = list(open(filename))

    # Split the proposal file into many parts which contain one video's
    # information separately.
    groups = groupby(lines, lambda x: x.startswith('#'))

    video_infos = [[x.strip() for x in list(g)] for k, g in groups if not k]

    def parse_group(video_info):
        """Parse the video's information.

        Template information of a video in a standard file:
            # index
            video_id
            num_frames
            fps
            num_gts
            label, start_frame, end_frame
            label, start_frame, end_frame
            ...
            num_proposals
            label, best_iou, overlap_self, start_frame, end_frame
            label, best_iou, overlap_self, start_frame, end_frame
            ...

        Example of a standard annotation file:

        .. code-block:: txt

            # 0
            video_validation_0000202
            5666
            1
            3
            8 130 185
            8 832 1136
            8 1303 1381
            5
            8 0.0620 0.0620 790 5671
            8 0.1656 0.1656 790 2619
            8 0.0833 0.0833 3945 5671
            8 0.0960 0.0960 4173 5671
            8 0.0614 0.0614 3327 5671

        Args:
            video_info (list): Information of the video.

        Returns:
            tuple[str, int, list, list]:
                video_id (str): Name of the video.
                num_frames (int): Number of frames in the video.
                gt_boxes (list): List of the information of gt boxes.
                proposal_boxes (list): List of the information of
                    proposal boxes.
        """
        offset = 0
        video_id = video_info[offset]
        offset += 1

        num_frames = int(float(video_info[1]) * float(video_info[2]))
        num_gts = int(video_info[3])
        offset = 4

        gt_boxes = [x.split() for x in video_info[offset:offset + num_gts]]
        offset += num_gts
        num_proposals = int(video_info[offset])
        offset += 1
        proposal_boxes = [
            x.split() for x in video_info[offset:offset + num_proposals]
        ]

        return video_id, num_frames, gt_boxes, proposal_boxes

    return [parse_group(video_info) for video_info in video_infos]


def perform_regression(detections):
    """Perform regression on detection results.

    Args:
        detections (list): Detection results before regression.

    Returns:
        list: Detection results after regression.
    """
    starts = detections[:, 0]
    ends = detections[:, 1]
    centers = (starts + ends) / 2
    durations = ends - starts

    new_centers = centers + durations * detections[:, 3]
    new_durations = durations * np.exp(detections[:, 4])

    new_detections = np.concatenate(
        (np.clip(new_centers - new_durations / 2, 0,
                 1)[:, None], np.clip(new_centers + new_durations / 2, 0,
                                      1)[:, None], detections[:, 2:]),
        axis=1)
    return new_detections


def temporal_nms(detections, threshold):
    """Parse the video's information.

    Args:
        detections (list): Detection results before NMS.
        threshold (float): Threshold of NMS.

    Returns:
        list: Detection results after NMS.
    """
    starts = detections[:, 0]
    ends = detections[:, 1]
    scores = detections[:, 2]

    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ious = temporal_iou(starts[order[1:]], ends[order[1:]], starts[i],
                            ends[i])
        idxs = np.where(ious <= threshold)[0]
        order = order[idxs + 1]

    return detections[keep, :]