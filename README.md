# NRSER
This repo is the implementation of the paper:  
Noise robust speech emotion recognition with signal-to-noise ratio adapting speech enhancement [[Arxiv]](https://arxiv.org/abs/2309.01164)


## Dataset:

Speech emotion dataset:  
[MSP-PODSCAST](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html) Database Release 1.10 (May 3, 2022) from The University of Texas at Dallas, Multimodal Signal Processing (MSP) Laboratory

Background noise dataset:  
[Audioset](https://research.google.com/audioset/)  
The training, validation, and testing wavfiles list are in audioset-train-80.txt, audioset-train-20.txt, audioset-val.txt.   
The excluded labels of environmental noise experiment is in human_generated_noise.csv.   
Note: some youtube videos were not available when we downloaded the data, see the above lists for the files that used in this study.    

## Data preprocessing & preparation

(1) Generate the enhanced signals of all training data (save time for model training). 

```
python CMGAN/enhanced_speech_cpu.py --test_dir /path/to/wavfiles/dir #if you use cpu
python CMGAN/enhanced_speech_gpu.py --test_dir /path/to/wavfiles/dir #if you use gpu
```
The enhanced signals will be saved in the ./data/{dir}\_en directory.

See [CMGAN](https://github.com/ruizhecao96/CMGAN) for more details. 

(2) Prepare the training txtfile

For SNR-level detection: 
- noise_train.txt
- noise_val.txt

```
wavpath_1; snr_level_1
wavpath_2; snr_level_2
e.g.
audio/clean_sampleA.wav; 1
audio/clean_sampleB.wav; 1
noise/_-lXVZ9QpO8.wav; 0
noise/_0bOQtWbqVc.wav; 0
```
Note: a sample that all values are 0 will cause error during training.  
 
For emotion recognition:
- emotion_train.txt
- emotion_val.txt

```
wavpath_1; emotion_category_1; A:arousal_1; V:valence_1; D:dominance_1;
wavpath_2; emotion_category_2; A:arousal_2; V:valence_2; D:dominance_2;
e.g. 
audio_noisy/noisy_sampleA.wav; N; A:4.500000; V:4.500000; D:5.000000;
audio_noisy/noisy_sampleB.wav; N; A:4.500000; V:4.500000; D:5.000000;
audio/clean_sampleA.wav; N; A:4.500000; V:4.500000; D:5.000000;
audio/clean_sampleB.wav; N; A:4.500000; V:4.500000; D:5.000000;
```

Note: for each sample in the datalist, there must be a corresponding enhanced signal with the same name in "{dir}\_en" directory

For example:   
audio_noisy_en/noisy_sampleA.wav 
#enhanced wav of "audio_noisy/noisy_sampleA.wav"  

audio_en/clean_sampleA.wav
#enhanced wav of "audio/clean_sampleA.wav"  

(3) Download SSL model 

Download huber_based_ls960.pt: [HuBERT Base](https://github.com/facebookresearch/fairseq/blob/main/examples/hubert/README.md)

## Source code of NRSER

Training:
```
python snr_model.py                                                           #Training phase1: training of SNR-level detection block
python emotion_model.py  --fairseq_base_model /path/to/hubert_base_ls960.pt   #Training phase2: training of emotion recognition block
python nrser.py --fairseq_base_model /path/to/hubert_base_ls960.pt            #Training phase3: fine-tuning the model
```

Testing - emotion recognition:
```
e.g.
python test_gpu.py --datadir ./test_samples --ckptdir emotion_model_s2_ccc2_r1-snr_model_r1 #if you use gpu
python test_cpu.py --datadir ./test_samples --ckptdir emotion_model_s2_ccc2_r1-snr_model_r1 #if you use cpu
```

Testing - SNR level detection (use the SNR-level detector without fine-tuning with emotion recognition.)
```
e.g.
python test_snr.py --datadir ./test_samples --ckptdir snr_model_r1 
```

## Evaluation code
evaluation_metric.py # to calculate Concordance Correlation Coefficient. 

## Pretrain model
[Google Drive](https://drive.google.com/drive/folders/1cBo7ZOAGPbqZUcsZuGuJflCELpFEirM9?usp=sharing)

## Citation
If you use the code in your research, please cite:  
....
Thanks :-)!

## License
* The NRSER work is released under MIT License. See LICENSE for more details.

## Acknowledgments
* [Speech Lab](http://www.cs.columbia.edu/speech/lab.cgi), CS, Columbia University, New York, United States
* [Bio-ASP Lab](https://bio-asplab.citi.sinica.edu.tw), CITI, Academia Sinica, Taipei, Taiwan
