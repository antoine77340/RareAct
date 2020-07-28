# RareAct

This repository contains annotation for the RareAct dataset as well as an evaluation script for
computing the wAP and sAP metrics descrivbed in the paper.

![RareAct](rareact.png)


## Data

You can download the videos zipped into one file [here](https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/howto100m_captions.zip)
The video names are the YouTube ids of the videos.

The annotation file is hosted in the github repo and named as rareact.csv

Here is a description of each column:


| Column Name           | Type                       | Example        | Description                                                                   |
| --------------------- | -------------------------- | -------------- | ----------------------------------------------------------------------------- |
| `id`                  | int                        | `14`           | Unique ID for the annotated video segment.                                    |
| `video_id`            | string                     | `7frRY7aGwMU`  | YouTube ID of the video where the segment originated from (unique per video). |
| `start`               | int                        | `3`            | Start time in seconds of the action segment.                                  |
| `end`                 | int                        | `5`            | End time in seconds of the action segment.                                    |
| `class_id`            | int                        | `8`            | The class identifier of the actions (verb, noun). Maximum id: .               |
| `verb`                | string                     | `cut`          | Action verb describing the interaction.                                       |
| `noun`                | string                     | `laptop`       | Object noun subject of the interaction.                                       |
| `annotation`          | int [0-4]                  | `1`            | Annotation for the given clip and (verb, noun) class. 1: Positive. 2: Hard negative (only verb is right): 3: Hard negative (only noun is right). 4: Hard negative (Both verb and noun are valid but verb is not applied to noun). 0: Negative.|


## Evaluation script

We provide an evaluation python script.
To run an evaluation you need first to create a prediction output numpy matrix of shape `7610x149`.
where each row represent the samples ordered similarly as in rareact.csv and each column is the prediction score for each of the action `class_id`.

To compute the mWAP just run:

```sh
python compute_score.py predictions.npy 
```

To compute the mSAP (n=100) just run:


```sh
python compute_score.py predictions.npy 100 
```

where predictions.npy is the prediction output numpy array as described above.

## References

If you find this dataset useful, please cite the following paper:

```bibtex
@article{miech20rareact,
   title={RareAct: A video dataset of unusual interactions},
   author={Miech, Antoine and Alayrac, Jean-Baptiste  and Laptev, Ivan and Sivic, Josef and Zisserman, Andrew},
   journal={arxiv},
   year={2020},
}
```
