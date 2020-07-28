# RareAct
TODO include image here.
This repository contains annotation for the RareAct dataset as well as an evaluation script for
computing the wAP and sAP metrics descrivbed in the paper.

## Data

You can download the videos zipped into one file [here](https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/howto100m_captions.zip)
The video names are the YouTube ids of the videos.

The annotation file is hosted in the github repo and named as rareact.csv

Here is a description of each column:

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">id</th>
<th valign="bottom">video\_id</th>
<th valign="bottom">start</th>
<th valign="bottom">end</th>
<th valign="bottom">class\_id</th>
<th valign="bottom">verb</th>
<th valign="bottom">noun</th>
<th valign="bottom">annotation</th>
<!-- TABLE BODY -->
<tr><td align="left">Unique identifier for the annotation</td>
<td align="center">YouTube id of the video</td>
<td align="center">The start time of the annotated clip</td>
<td align="center">The end time of the annotated clip</td>
<td align="center">The class_id of the actions (verb, noun). Maximum id: . Type: int. </td>
<td align="center">Action verb describing class_id. Type: string</td>
<td align="center">Object noun describing class_id. Type: string</td>
<td align="center">Annotation for the given clip and (verb, noun) class. 1: Positive. 2: Hard negative (only verb is right): 3: Hard negative (only noun is right). 4: Hard negative (Both verb and noun are valid but verb is not applied to noun). 0: Negative. Type: integer in [0,1,2,3,4]</td>
</tr>
</tbody></table>

## Evaluation script

We provide an evaluation python script.
To run an evaluation you need first to create a prediction output numpy matrix of shape 901x144.
where each row represent the samples ordered similarly as in rareact.csv and each column is the prediction score for each of the action class_id.

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
