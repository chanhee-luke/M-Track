from collections import OrderedDict
import copy
import itertools
import json
import numpy as np
import os
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask_util
from tabulate import tabulate

from detectron2.evaluation.coco_evaluation import COCOEvaluator, _evaluate_predictions_on_coco
from detectron2.evaluation.fast_eval_api import COCOeval_opt
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table


class AlfredEvaluator(COCOEvaluator):
    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_alfred_json(instances, input["image_id"])
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            if len(prediction) > 1:
                self._predictions.append(prediction)

    def _eval_predictions(self, predictions, img_ids=None):
        """
        Evaluate predictions. Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        tasks = self._tasks or self._tasks_from_predictions(coco_results)

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
            all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
            num_classes = len(all_contiguous_ids)
            assert min(all_contiguous_ids) == 0 and max(all_contiguous_ids) == num_classes - 1

            reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
            for result in coco_results:
                category_id = result["category_id"]
                assert category_id < num_classes, (
                    f"A prediction has class={category_id}, "
                    f"but the dataset only has {num_classes} classes and "
                    f"predicted class id should be in [0, {num_classes - 1}]."
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info(
            "Evaluating predictions with {} COCO API...".format(
                "unofficial" if self._use_fast_impl else "official"
            )
        )
        coco_evals = {}
        for task in sorted(tasks):
            assert task in {"bbox", "segm", "keypoints"}, f"Got unknown task: {task}!"
            coco_eval = (
                _evaluate_predictions_on_coco(
                    self._coco_api,
                    coco_results,
                    task,
                    kpt_oks_sigmas=self._kpt_oks_sigmas,
                    use_fast_impl=self._use_fast_impl,
                    img_ids=img_ids,
                )
                if len(coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )

            res = self._derive_coco_results(
                coco_eval, task, class_names=self._metadata.get("thing_classes")
            )
            self._results[task] = res
            coco_evals[task] = coco_eval

        coco_eval = coco_evals["bbox"]
        rch_results = self._eval_reachable(
            coco_eval,
            class_names=self._metadata.get("thing_classes"),
        )
        self._results["rch"] = rch_results

    def _eval_reachable(self, coco_eval, class_names=None):
        coco_eval = copy.deepcopy(coco_eval)
        p = coco_eval.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]

        # fmt: off
        # T           = len(p.iouThrs)
        # R           = len(p.recThrs)
        # K           = len(p.catIds) if p.useCats else 1
        # A           = len(p.areaRng)
        # M           = len(p.maxDets)
        # fmt: on

        # create dictionary for future indexing
        _pe = coco_eval._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        # setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds) if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        # a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds) if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        maxDet = m_list[-1]

        counts, prs, rcs, accs, tnrs = [], [], [], [], []
        for k, k0 in enumerate(k_list):
            Nk = k0 * A0 * I0
            # for a, a0 in enumerate(a_list):
            # Na = a0 * I0
            # for m, maxDet in enumerate(m_list):
            E = [coco_eval.evalImgs[Nk + i] for i in i_list]
            E = [e for e in E if e is not None]
            if len(E) == 0:
                for mc in [prs, rcs, accs, tnrs]:
                    mc.append(float("nan"))
                counts.append((0, 0))
                continue
            dtScores = np.concatenate([e["dtScores"][0:maxDet] for e in E])

            # different sorting method generates slightly different results.
            # mergesort is used to be consistent as Matlab implementation.
            inds = np.argsort(-dtScores, kind="mergesort")
            # dtScoresSorted = dtScores[inds]

            dtm = np.concatenate([e["dtMatches"][0, 0:maxDet] for e in E]).astype(int)[inds]
            dtIds = (
                np.concatenate([e["dtIds"][0:maxDet] for e in E])
                .astype(int)[inds][dtm > 0]
                .tolist()
            )
            gtIds = dtm[dtm > 0].tolist()
            # gtIds = np.concatenate([e["gtIds"][0:maxDet] for e in E])[inds][dtm >= 0].tolist()

            dts = coco_eval.cocoDt.loadAnns(dtIds)
            gts = coco_eval.cocoGt.loadAnns(gtIds)

            dt_reachable = np.array([dt["reachable"] for dt in dts]) > 0.5
            gt_reachable = np.array([gt["reachable"] for gt in gts], dtype=bool)

            tp = np.logical_and(gt_reachable, dt_reachable).sum()
            pr = tp / (dt_reachable.sum() + np.spacing(1))
            rc = tp / (gt_reachable.sum() + np.spacing(1))
            acc = (gt_reachable == dt_reachable).sum() / (len(dt_reachable) + np.spacing(1))

            tn = np.logical_and(~gt_reachable, ~dt_reachable).sum()
            tnr = tn / ((~gt_reachable).sum() + np.spacing(1))

            prs.append(pr)
            rcs.append(rc)
            accs.append(acc)
            tnrs.append(tnr)

            counts.append((len(gt_reachable), int((gt_reachable == dt_reachable).sum())))

        results = OrderedDict(
            {
                "rch_mP50": np.nanmean(prs) * 100,
                "rch_mTPR50": np.nanmean(rcs) * 100,
                "rch_mTNR50": np.nanmean(tnrs) * 100,
                "rch_mACC50": np.nanmean(accs) * 100,
            }
        )

        if class_names is not None:
            assert len(class_names) == len(
                prs
            ), f"{len(class_names)} classes but {len(prs)} precissions got"
            results_per_category = []
            for idx, name in enumerate(class_names):
                results_per_category.append(("{}".format(name), float(prs[idx] * 100)))

            if self._output_dir:
                file_path = os.path.join(self._output_dir, "alfred_per_category.json")
                self._logger.info("Saving Alfred per-category results to {}".format(file_path))
                with PathManager.open(file_path, "w") as f:
                    f.write(json.dumps({"P50": prs, "TPR50": rcs, "TNR50": tnrs, "ACC50": accs}))
                    f.flush()

                file_path = os.path.join(self._output_dir, "alfred_counts_per_category.json")
                with PathManager.open(file_path, "w") as f:
                    f.write(json.dumps(counts))
                    f.flush()

            N_COLS = min(6, len(results_per_category) * 2)
            results_flatten = list(itertools.chain(*results_per_category))
            results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
            table = tabulate(
                results_2d,
                tablefmt="pipe",
                floatfmt=".3f",
                headers=["category", "AP"] * (N_COLS // 2),
                numalign="left",
            )
            self._logger.info("Per-category Reachable Precision: \n" + table)

        return results

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.
        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.
        Returns:
            a dict of {metric name: score}
        """

        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))
        if self._output_dir:
            file_path = os.path.join(self._output_dir, f"coco_{iou_type}_per_category.json")
            self._logger.info(
                "Saving COCO per-category {} results to {}".format(iou_type, file_path)
            )
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(results_per_category))
                f.flush()

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        results.update({"AP-" + name: ap for name, ap in results_per_category})
        return results


def instances_to_alfred_json(instances, img_id):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.
    Args:
        instances (Instances):
        img_id (int): the image id
    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()
    reachables = instances.pred_reachables.tolist()

    has_mask = instances.has("pred_masks")
    if has_mask:
        # use RLE to encode the masks, because they are too large and takes memory
        # since this evaluator stores outputs of the entire dataset
        rles = [
            mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            for mask in instances.pred_masks
        ]
        for rle in rles:
            # "counts" is an array encoded by mask_util as a byte-stream. Python3's
            # json writer which always produces strings cannot serialize a bytestream
            # unless you decode it. Thankfully, utf-8 works out (which is also what
            # the pycocotools/_mask.pyx does).
            rle["counts"] = rle["counts"].decode("utf-8")

    has_keypoints = instances.has("pred_keypoints")
    if has_keypoints:
        keypoints = instances.pred_keypoints

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
            "reachable": reachables[k],
        }
        if has_mask:
            result["segmentation"] = rles[k]
        if has_keypoints:
            # In COCO annotations,
            # keypoints coordinates are pixel indices.
            # However our predictions are floating point coordinates.
            # Therefore we subtract 0.5 to be consistent with the annotation format.
            # This is the inverse of data loading logic in `datasets/coco.py`.
            keypoints[k][:, :2] -= 0.5
            result["keypoints"] = keypoints[k].flatten().tolist()
        results.append(result)
    return results
