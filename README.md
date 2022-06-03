# Differentiable Paint Transformer
## Usage

Generate strokes
```
python .\image2storkes.py --input_path ./input/bingbing.jpg --output_dir ./output --use_neural_render
```

Stylize strokes
```
 python .\strokes2stylized.py --input_path ./input/bingbing.jpg --output_dir ./output --use_mask --style_path ./input/mosaic.jpg
```