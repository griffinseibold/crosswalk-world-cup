# Data

What data the project uses, the rules it follows for footage of people,
where the data lives, and how to label the fixed-camera evaluation set.

## Footage of public spaces

The workload is recorded road footage: intersections, crosswalks, traffic.
Footage like that shows people who did not agree to be in a dataset. These
rules apply to every clip, whoever recorded it:

1. **Recorded footage only.** The code has no live-camera, RTSP or webcam
   input, and adding one is out of scope. A live feed of a public place is
   surveillance whatever the intent; recorded clips can be reviewed, blurred
   and deleted before any processing.
2. **Blur faces and licence plates before storing.** Clips are blurred on the
   machine that recorded them, before upload to `s3://datasets/footage/` or
   anywhere else. Only blurred clips enter the cluster. Tools that do this:
   - [deface](https://github.com/ORB-HD/deface): face blurring from the
     command line (`deface clip.mp4`);
   - [EgoBlur](https://github.com/facebookresearch/EgoBlur): faces and licence
     plates.

   Check the output by eye. Automatic blurring misses small, distant and
   partly hidden faces, which is exactly what a wide intersection shot is
   full of.
3. **Blurring does not grant permission.** A blurred clip can still identify
   someone (by clothing, gait, location and time), and it is still footage of
   them. Blurring reduces harm; it does not make a clip yours to use. Before
   recording, check the law where you are, any rules of the property you
   record from or on, and the terms of anything you did not record yourself.
   If in doubt, do not record.
4. **Keep only what is needed.** Detection output stores boxes, labels and
   timestamps, not images. Delete raw clips once they are blurred and checked.
   Delete blurred clips when they are no longer used.
5. **No re-identification.** Nothing in this project tries to recognise,
   track across clips, or link a detection to a specific person or vehicle.
   The tracker only follows a box from frame to frame within one
   clip to count crossings.

Footage never goes in Git. `.gitignore` excludes common video formats, but
the rule is not to put it in the repository folder at all.

## Datasets

Openly licensed candidates for footage and training data (fixed-camera
intersections, BDD100K alternatives), with what was verified about each
licence: [data sources](data-sources.md).

| Dataset | Use | Licence | How it gets here |
| --- | --- | --- | --- |
| COCO 2017 (via TorchVision weights) | The pretrained detector and baseline | Weights: BSD-3 (TorchVision); COCO annotations: CC BY 4.0 | TorchVision downloads the weights on first use |
| [BDD100K](https://bdd-data.berkeley.edu/) | Fine-tuning on road scenes | Non-commercial research licence; registration required | You download it, accept the licence and upload it yourself; the code only reads a local path |
| Fixed-camera eval set | Scoring both models on this project's real viewpoint | Your own footage, under the rules above | Recorded, blurred, labeled by you (below) |
| Synthetic | Tests and smoke runs | Generated (`crosswalk make-synthetic`) | Generated on demand |

### BDD100K

The code never downloads BDD100K. Its licence restricts it to
non-commercial use and requires registering and agreeing to the terms, and
an automated download would sidestep that. Do it once, by hand:

1. Register at <https://bdd-data.berkeley.edu/> and read the licence. This
   project is non-commercial, but anything sold or offered as a service that
   is built on the fine-tuned weights may not be. Keep the
   fine-tuned model to non-commercial use, or retrain on data you have
   commercial rights to.
2. Download the **100K images** and the **detection labels (det_20)**:
   `det_train.json` and `det_val.json`.
3. Upload them to the datasets bucket, keeping this layout (the paths in
   `configs/train-bdd100k.yaml` assume it):

   ```text
   s3://datasets/bdd100k/images/100k/train/*.jpg
   s3://datasets/bdd100k/images/100k/val/*.jpg
   s3://datasets/bdd100k/labels/det_20/det_train.json
   s3://datasets/bdd100k/labels/det_20/det_val.json
   ```

   For example, from the host with the AWS CLI and the `s3-credentials`
   keys: `aws --endpoint-url <s3 endpoint> s3 sync ./bdd100k s3://datasets/bdd100k`.

The training Job copies `s3://datasets/bdd100k/` to its scratch volume
before training. `data.train.max_images` picks a reproducible subset (same
seed, same images), so a full 70k-image download is not needed to start.
Both det_20 and the older 2018 `bdd100k_labels_images_*.json` files parse.
Lane and drivable-area polygons are skipped, since only boxes are used.

The class mapping lives in the config (`data.*.category_map`). For example,
BDD100K's `rider` is counted as `pedestrian`. Categories that map to no
project class are dropped and counted, and the counts are logged with the
dataset manifest, so a mapping mistake is visible.

### Dataset hashes

Every training run records a content hash of its training and eval sets in
MLflow (tags `dataset.sha256`, `eval_dataset.sha256`; full manifests under
`dataset/` in the run's artifacts). The hash covers:

- the bytes of the labels files;
- the parsed, selected samples: a different subset, category map or
  attribute filter changes it;
- the bytes of every selected image (`data.hash_images`, on by default), so
  re-blurring or re-encoding an image changes it.

Two runs with the same `dataset.sha256` trained on exactly the same data.
`crosswalk hash-dataset --config <file>` prints the manifests without
training.

## The fixed-camera evaluation set

About 200 frames from the camera position the project will actually use.
BDD100K is dashcam footage at street level, so a model can do well on its
validation set and badly on a fixed camera looking down at a crosswalk. This
set measures what matters: is the fine-tuned model better than the
pretrained one *from here*?

### Choosing frames

- From blurred clips only.
- Spread across times of day, weather and traffic levels; not 200 frames
  from one minute.
- At least a few seconds apart, so neighbouring frames are not near-copies.
- From clips never used for training. Keep a list of which clips went into
  the eval set.

Extract frames with FFmpeg, for example one frame every 10 seconds:

```bash
ffmpeg -i blurred-clip.mp4 -vf fps=1/10 -q:v 2 frames/clip01-%04d.jpg
```

### Labeling

Use [CVAT](https://www.cvat.ai/) or [Label Studio](https://labelstud.io/).
Both are open source and can run locally (CVAT with Docker Compose; Label
Studio with `pip install label-studio`). Either can also run on the cluster.

1. Create a project with exactly the project's class names, spelled as in
   the config: `car`, `pedestrian`, `truck`, `bus`, `bicycle`, `motorcycle`,
   `traffic light`. If your tool uses `person`, add `person: pedestrian` to
   the eval set's `category_map`.
2. Draw tight boxes around every visible instance, including small and
   partly hidden ones. Unlabeled objects count against the model as false
   positives.
3. Mark dense crowds where single boxes are impossible as crowd regions
   (`iscrowd`). Evaluation then ignores detections inside them instead of
   penalising them.
4. Write a one-page labeling guide (what counts as a pedestrian on a bike,
   when to box a half-visible car) and follow it. Consistency matters more
   than any particular rule.
5. Export:
   - CVAT: **Export dataset → COCO 1.0**, giving `annotations/instances_default.json`.
   - Label Studio: **Export → COCO**, giving `result.json` plus `images/`.

6. Upload, keeping file names as exported:

   ```text
   s3://datasets/fixed-camera-eval/images/*.jpg
   s3://datasets/fixed-camera-eval/annotations/instances_default.json
   ```

   Then point `data.eval.labels` and `data.eval.images` at those paths
   under `/scratch/data/fixed-camera-eval/`, as in `configs/train-bdd100k.yaml`.

Check the export before relying on it:

```bash
crosswalk hash-dataset --config configs/train-bdd100k.yaml
```

The `eval` manifest shows the image count, the boxes per class, and any
dropped categories.

### Scoring

At the end of training, both the fine-tuned model and the pretrained COCO
model are scored on this set with pycocotools (COCO mAP@[.5:.95], mAP@.5,
per-class AP). The pretrained model predicts COCO categories. They are mapped
to project classes by name (`eval.baseline_category_map`, e.g. `person` →
`pedestrian`), and COCO classes the project does not label are discarded
before scoring. To score any checkpoint later:

```bash
crosswalk evaluate --config configs/train-bdd100k.yaml --checkpoint s3://checkpoints/<run>/step-00001500.pt
```
