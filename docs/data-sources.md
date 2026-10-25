# Data sources

Candidate footage and training data that are openly licensed and need no
registration, researched in October 2026. **No video is committed to this
repository** (`.gitignore` blocks common video formats). Each source is
downloaded by hand, checked, blurred if needed ([data handling](data.md)),
and uploaded to `s3://datasets/footage/`.

Licences change and pages move. Every row below records what was checked,
where, and what could **not** be verified. Read the licence at the source
again before downloading anything, and keep a copy of it next to the data.

## What the project needs

1. **Fixed-camera footage of crossings** (the leaderboard's input): a
   stationary camera overlooking a crosswalk, with pedestrians and vehicles,
   long enough for at least 10 minutes per crossing (the ranking minimum).
2. **Training data with boxes** for fine-tuning, to replace or complement
   BDD100K, whose licence is non-commercial and needs registration.

## Fixed-camera footage

| Source | What | Licence (as verified) | Registration | Verdict |
| --- | --- | --- | --- | --- |
| [NGSIM Lankershim Boulevard](https://www.fhwa.dot.gov/publications/research/operations/07029/index.cfm) (US DOT FHWA) | Five cameras on the roof of a 36-storey building over Lankershim Blvd, Los Angeles. Three signalised intersections over about 500 m, 16 June 2005, 8:30–9:00 a.m. | The NGSIM *trajectory* dataset on data.gov lists [CC BY-SA 3.0](https://catalog.data.gov/dataset/next-generation-simulation-ngsim-vehicle-trajectories-and-supporting-data), access level "public". The separate **video** datasets ("NGSIM Lankershim Boulevard Videos", "NGSIM Peachtree Street Videos") are listed from that page, but their own licence pages could not be fetched from this environment. **Unverified** | None found | **Best candidate.** Fixed, elevated, real intersections, a US government programme. Confirm the video datasets' licence on data.transportation.gov before use. FHWA's report does not mention pedestrians, so check that they are visible at the resolution provided. Footage from 2005 is "recorded" in every sense; the leaderboard says so |
| NGSIM Peachtree Street (Atlanta) | Same programme: cameras over an arterial with intersections | As above. **Unverified** | None found | Second NGSIM candidate, same caveats |
| [Shibuya Crossing, Tokyo (video)](https://c.enwp.org/wiki/File:Shibuya_Crossing,_Tokyo,_Japan_(video).webm), Wikimedia Commons | 59 s, 1920×1080, by Basile Morin | **CC BY-SA 4.0** (verified on the file page) | None | Good licence and a famous crossing, but only 59 s (below the 10-minute ranking minimum), and the page does not say whether the camera is fixed. Useful for a demo or a detector sanity check, not for a ranked crossing. Attribution: "Basile Morin, CC BY-SA 4.0", with a link to the licence and a note of any changes (for example "blurred, resized"). Share-alike applies to derived *video*, not to the counts computed from it |
| [Xiph.org test media](https://media.xiph.org/video/derf/) (`pedestrian_area`, `rush_hour`) | Short 1080p uncompressed test sequences, 375 and 500 frames. Also on [AWS Open Data](https://registry.opendata.aws/xiph-media), anonymous S3 | The collection says "believed to be freely redistributable… see the appropriate copyright file". AWS lists the licence as "Various. Most are under the CC-BY license". The per-sequence readme files could not be fetched here (robots.txt). **Unverified per sequence** | None (anonymous S3) | 15–20 s each: too short to rank, but handy for benchmark clips (Experiment A) once each sequence's own licence file is read |
| [StreetAware](https://ultraviolet.library.nyu.edu/records/q1byv-qc065) (NYU) | About 8 h of synchronised, stationary, high-resolution video at three Brooklyn intersections, two sensors per side of each crosswalk, faces blurred | The paper is CC BY. The dataset record states **no licence** that could be found, and the ~550 GB of data is delivered through Globus | Globus transfer (an account) | Ideal content, but unusable until a data licence is confirmed with the authors. Do not use on the strength of the paper's licence |
| [MIO-TCD](https://tcd.miovision.com/) | Over half a million images from 8,000 traffic cameras in Canada and the US | **CC BY-NC-SA 4.0** (verified) | Not stated | Non-commercial and images rather than video: fine for detector evaluation in a portfolio project, not for continuous footage |
| [Ann Arbor intersection trajectories](https://github.com/michigan-traffic-lab/Ann-Arbor-Intersection-Trajectory-Data) | Roadside-perception **trajectories** (JSON), no video | README says CC BY-SA 4.0 but links the BY-NC-SA licence: **ambiguous** | None | Not footage. Could serve as a reference for indicator ranges |

**Recommendation:** start with NGSIM Lankershim once the video licence is
confirmed on data.transportation.gov. `configs/cameras/ngsim-lankershim.yaml`
is its camera config, uncalibrated until someone picks the four reference
points. Your own recordings, made under [the rules in data.md](data.md), are
the other reliable source.

## Training data without registration (BDD100K alternatives)

| Dataset | What | Licence (as verified) | Registration | Notes |
| --- | --- | --- | --- | --- |
| [SHIFT](https://www.vis.xyz/shift/) (ETH Zürich) | Synthetic driving (CARLA), 2D/3D boxes for cars and pedestrians, among others | **CC BY-SA 4.0**: "free for academic and non-academic use for research, teaching, publications, and personal experimentation" (verified on the dataset page). The dev kit is MIT | None stated; the [download script](https://github.com/SysCV/shift-dev) fetches from `dl.cv.ethz.ch` directly | Synthetic, so domain shift to real cameras. Good for pre-training and for checking the training platform at scale without any licence question. Front-view `img` + `det_2d` at 1 fps is the subset to start with |
| [Zenseact Open Dataset (ZOD)](https://huggingface.co/datasets/Zenseact/ZOD) | Real driving from 14 European countries, 2D boxes, faces and plates anonymised by a third party | **CC BY-SA 4.0**, research and commercial use (verified in the [ICCV 2023 paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Alibeigi_Zenseact_Open_Dataset_A_Large-Scale_and_Diverse_Multimodal_Dataset_for_ICCV_2023_paper.pdf) and the Hugging Face card) | The official route **needs an access application** (`zod` dev kit: "you have applied for access and received a download link"). The Hugging Face copy shows no gate on its card, which could not be confirmed | The closest real-world, openly licensed replacement for BDD100K. Any public display must carry Zenseact's privacy notice ([zod-web](https://github.com/zenseact/zod-web)). Dashcam viewpoint, like BDD100K |
| COCO 2017 | Already used: the pretrained detector's training data | Annotations CC BY 4.0. Images carry their Flickr licences | None | Unchanged |

None of these replace the **fixed-camera evaluation set**. Driving datasets
are filmed from a car, and the leaderboard watches from a pole. The
hand-labelled set ([data.md](data.md#the-fixed-camera-evaluation-set))
remains the only fair test of a model for this viewpoint.

## Attribution on the leaderboard

Each camera config carries `source`, `license` and `attribution`. The ranking
API returns them with every crossing, so the page can credit the footage
next to its numbers. For CC BY and CC BY-SA sources, fill in all three before
the camera is deployed.

## Not used, and why

- **Live camera feeds** (city traffic cameras, 511 streams, webcams): out of
  scope by design, whatever the licence ([data.md](data.md)).
- **Stock footage sites** (free or paid): their licences usually forbid
  redistribution and datasets, and they are not open licences.
- **Scraped video** (YouTube and similar): no licence to process or
  redistribute.

## What was checked from this environment

Verified by reading the source page: the Shibuya clip's licence, MIO-TCD's
licence, SHIFT's licence and the MIT dev kit, ZOD's licence and access
requirement, the NGSIM trajectory dataset's licence field, FHWA's description
of the Lankershim camera setup, the Xiph collection's general terms and its
AWS listing, and StreetAware's record and README (no data licence found).
Not verifiable here: the NGSIM *video* datasets' own licence pages and the
Xiph per-sequence readmes (both blocked), and whether the ZOD Hugging Face
copy is gated.
