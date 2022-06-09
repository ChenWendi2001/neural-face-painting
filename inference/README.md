# Inference:hugs:
## Models
Download the [checkpoints](https://jbox.sjtu.edu.cn/l/R1nghh) and unzip it to the `./checkpoints` folder.

## Usage

1. Generate strokes

   ```python
   python ./image2storkes.py --input_path ./input/bingbing.jpg --output_dir ./output --use_neural_render --strokes_increasing_layers 1
   ```

2. Stylize strokes

   ```python
   python ./strokes2stylized.py --input_path ./input/bingbing.jpg --output_dir ./output --use_mask --style_path ./input/mosaic.jpg
   ```

   