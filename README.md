# Mono-DDGS
Mono-DDGS: Endoscopic Monocular Scene Reconstruction with Dual-Stage Decoupled Gaussian Splatting

## Environment setup
```bash
git clone https://github.com/bawangxiaoxia111/Mono-DDGS-main.git
cd Mono-DDGS
git submodule update --init --recursive
conda create -n Mono-DDGS python=3.8 
conda activate Mono-DDGS

pip install -r requirements.txt
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

## Datasets
**For monocular scenes**: Gastrolab Dataset could be downloaded from their [official websites](https://www.gastrolab.net/video.htm). For our experiments, we selected 47 consecutive frames (00102–00148) from the [video156](https://www.gastrolab.net/video156.htm) video sequences.

The [CholecSeg8k](https://arxiv.org/pdf/2012.12453) dataset can be downloaded from the publicly available repository on [Kaggle](https://www.kaggle.com/datasets/newslab/cholecseg8k). We selected 15 consecutive frames (frame_304_endo–frame_319_endo) from the *video 24* sequence for experiments.

**For stereo scenes**: We used the original [EndoNeRF](https://github.com/med-air/EndoNeRF) image sequences provided by [Yuehao Wang](https://docs.google.com/forms/d/e/1FAIpQLSfM0ukpixJkZzlK1G3QSA7CMCoOJMFFdHm5ltCV1K6GNVb3nQ/viewform). 

Camera poses for all datasets were obtained using COLMAP. The preprocessed dataset is available upon request. Please complete the [dataset request form](https://docs.google.com/forms/d/e/1FAIpQLSeuX2rgUSC0-g4EyS6pBaYeNTQiVIjrUhmLz26xifP_KlyLWA/viewform), and approved users will receive the Google Drive download link.

<!--We also provide the preprocessed dataset: [Google Drive](https://drive.google.com/file/d/1_hPQLEmtZ6AS2jnOIAFC8iPZ4Nqc8JKu/view?usp=sharing).-->
```text 
├── data
│   │ Gastrolab 
│     ├── vid156
│   │ CholecSeg8k 
│     ├── video24
│   │ EndoNeRF 
│     ├── Cutting
│     └── ...
```

## Training
For training scenes such cutting, run
```bash
python train.py -s data/dnerf/cutting --port 6017 --expname "dnerf/cutting" --configs arguments/dnerf/cutting.py
```

## Rendering
Run the following script to render the images.
```bash
python render.py --model_path "output/dnerf/cutting/" --skip_train --configs arguments/dnerf/cutting.py
```

## Specular Reflection Removal
cd ./En_EndoSRR
Download  our pretrained En-EndoSRR weights and place it in the pretrained folder.
Download the Big-LaMa pretrained model of LaMa and place it in the pretrained folder.

Run the following command to perform specular reflection removal on rendered images:

```bash
python En_EndoSRR.py --input_dir ./rendered_images --output_dir ./results --weights ./checkpoints/En-EndoSRR.pth
```

## Acknowledgements
We would like to acknowledge the following inspiring work:

[3DGS](https://github.com/graphdeco-inria/gaussian-splatting) (Bernhard Kerbl et al.)


