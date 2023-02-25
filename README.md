# NRSER
Code for the NRSER paper

## Dataset used in the study:

Speech emotion recognition:
MSP-PODSCAST Database Release 1.4 (Feb 10, 2019) from The University of Texas at Dallas, Multimodal Signal Processing (MSP) Laboratory
https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html

Audioset:
https://research.google.com/audioset/ 
Some youtube videos were not available when downloading the data. 
The training, validation, and testing wavfiles list are in audioset-train-80.txt, audioset-train-20.txt, audioset-val.txt. 
The excluded labels of environmental noise experiment is in human_generated_noise.csv. 

## Data preprocessing


## Source code of NRSER


```
python noise_model.py         #Training phase1: training of SNR-level detection block
python emotion_model.py       #Training phase2: training of emotion recognition block
python model_finetune.py.     #Training phase3: fine-tuning the model

```

```
python test.py 
```
## Evaluation code

evaluation_metric.py # code to calculate Concordance Correlation Coefficient. 



## Citation
If you use the code in your research, please cite:  
....
Thanks :-)!

## License
* The NRSER work is released under MIT License. See LICENSE for more details.

## Acknowledgments
* [Speech Lab] (http://www.cs.columbia.edu/speech/lab.cgi), CS, Columbia University, New York, United States
* [Bio-ASP Lab](https://bio-asplab.citi.sinica.edu.tw), CITI, Academia Sinica, Taipei, Taiwan
