# NRSER
Code for the NRSER paper

## Dataset used in the study:

Speech emotion recognition:
[MSP-PODSCAST] (https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html) Database Release 1.4 (Feb 10, 2019) from The University of Texas at Dallas, Multimodal Signal Processing (MSP) Laboratory

Background noise dataset:
[Audioset](https://research.google.com/audioset/)
The training, validation, and testing wavfiles list are in audioset-train-80.txt, audioset-train-20.txt, audioset-val.txt.  
The excluded labels of environmental noise experiment is in human_generated_noise.csv.  
Note: some youtube videos were not available when we downloaded the data, see the above lists for the files that used in this study.   

## Data preprocessing

(1) Generate the enhanced signals of all training data (save time for model training). 

```
python CMGAN/enhanced_speech_cpu.py --test_dir /path/to/wavfiles/dir #if you use cpu
python CMGAN/enhanced_speech_gpu.py --test_dir /path/to/wavfiles/dir #if you use gpu
```
The enhanced signals will be saved in the ./Results/{dir}\_en directory.

See [CMGAN](https://github.com/ruizhecao96/CMGAN) for more details. 

(2) Prepare the training txtfile

For SNR-level detection: 
- noise_train.txt
- noise_val.txt

```
wavpath_1; snr_level_1
wavpath_2; snr_level_2
e.g.
audio/MSP-PODCAST_0001; 1
Audioset-train-80; 0
```
Note: a sample that all values are 0 will cause error during training.  
 
For emotion recognition:
- emotion_train.txt
- emotion_val.txt

```
wavpath_1; emotion_category_1; A:arousal_1; V:valence_1; D:dominance_1;
wavpath_2; emotion_category_2; A:arousal_2; V:valence_2; D:dominance_2;
e.g. 
MSP_train_noisy/MSP-PODCAST_0001_0002_snr6_n53.wav; N; A:3.500000; V:3.500000; D:3.500000;
MSP_train_noisy/MSP-PODCAST_0001_0002_snr6_n36.wav; N; A:3.500000; V:3.500000; D:3.500000;
```

Note: for each sample in the datalist, there must be a corresponsing enhanced signal with the same name in "{dir}\_en" directory

For example:   
MSP_train_noisy_en/MSP-PODCAST_0001_0002_snr6_n53.wav   
#enhanced wav of "MSP_train_noisy/MSP-PODCAST_0001_0002_snr6_n53.wav"  

MSP_train_noisy_en/MSP-PODCAST_0001_0002_snr6_n36.wav  
#enhanced wav of "MSP_train_noisy/MSP-PODCAST_0001_0002_snr6_n36.wav"  

## Source code of NRSER

Training:
```
python noise_model.py         #Training phase1: training of SNR-level detection block
python emotion_model.py       #Training phase2: training of emotion recognition block
python model_finetune.py     #Training phase3: fine-tuning the model
```

Testing:
```
python test.py --datadir /path/to/test/wav --ckptdir /path/to/model/ckpt
e.g.
python test_gpu.py --datadir ./samples --ckptdir emotion_model_v1_audioset-noise_model_v1_audioset-f16 #if you use gpu
python test_cpu.py --datadir ./samples --ckptdir emotion_model_v1_audioset-noise_model_v1_audioset-f16 #if you use cpu
```
## Evaluation code

evaluation_metric.py # code to calculate Concordance Correlation Coefficient. 

## Pretrain model
[Google Drive](https://drive.google.com/drive/folders/12dTsiwFuPEu7n3tKJdSdko2-CfSvYlVz?usp=sharing) 

## Citation
If you use the code in your research, please cite:  
....
Thanks :-)!

## License
* The NRSER work is released under MIT License. See LICENSE for more details.

## Acknowledgments
* [Speech Lab](http://www.cs.columbia.edu/speech/lab.cgi), CS, Columbia University, New York, United States
* [Bio-ASP Lab](https://bio-asplab.citi.sinica.edu.tw), CITI, Academia Sinica, Taipei, Taiwan
