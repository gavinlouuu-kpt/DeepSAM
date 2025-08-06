<p float="center">
  <img src="assets/logo2.png?raw=true" width="99.1%" />
</p>

# Faster Segment Anything (MobileSAM) and Everything (MobileSAMv2)
:pushpin: MobileSAMv2, available at [ResearchGate](https://www.researchgate.net/publication/376579294_MobileSAMv2_Faster_Segment_Anything_to_Everything) and [arXiv](https://arxiv.org/abs/2312.09579.pdf), replaces the grid-search prompt sampling in SAM with object-aware prompt sampling for faster **segment everything(SegEvery)**.

:pushpin: MobileSAM, available at [ResearchGate](https://www.researchgate.net/publication/371851844_Faster_Segment_Anything_Towards_Lightweight_SAM_for_Mobile_Applications) and [arXiv](https://arxiv.org/pdf/2306.14289.pdf), replaces the heavyweight image encoder in SAM with a lightweight image encoder for faster **segment anything(SegAny)**. 


**Support for ONNX model export**. Feel free to test it on your devices and share your results with us.

**A demo of MobileSAM** running on **CPU** is open at [hugging face demo](https://huggingface.co/spaces/dhkim2810/MobileSAM). On our own Mac i5 CPU, it takes around 3s. On the hugging face demo, the interface and inferior CPUs make it slower but still works fine. Stayed tuned for a new version with more features! You can also run a demo of MobileSAM on [your local PC](https://github.com/ChaoningZhang/MobileSAM/tree/master/app).

:grapes: Media coverage and Projects that adapt from SAM to MobileSAM (Thank you all!)
* **2023/07/03**: [joliGEN](https://github.com/jolibrain/joliGEN) supports MobileSAM for faster and lightweight mask refinement for image inpainting with Diffusion and GAN.
* **2023/07/03**: [MobileSAM-in-the-Browser](https://github.com/akbartus/MobileSAM-in-the-Browser) shows a demo of running MobileSAM on the browser of your local PC or Mobile phone.
* **2023/07/02**: [Inpaint-Anything](https://github.com/qiaoyu1002/Inpaint-Anything) supports MobileSAM for faster and lightweight Inpaint Anything
* **2023/07/02**: [Personalize-SAM](https://github.com/qiaoyu1002/Personalize-SAM) supports MobileSAM for faster and lightweight Personalize Segment Anything with 1 Shot
* **2023/07/01**: [MobileSAM-in-the-Browser](https://github.com/akbartus/MobileSAM-in-the-Browser) makes an example implementation of MobileSAM in the browser.
* **2023/06/30**: [SegmentAnythingin3D](https://github.com/Jumpat/SegmentAnythingin3D) supports MobileSAM to segment anything in 3D efficiently.
* **2023/06/30**: MobileSAM has been featured by [AK](https://twitter.com/_akhaliq?lang=en) for the second time, see the link [AK's MobileSAM tweet](https://twitter.com/_akhaliq/status/1674410573075718145). Welcome to retweet.
* **2023/06/29**: [AnyLabeling](https://github.com/vietanhdev/anylabeling) supports MobileSAM for auto-labeling. 
* **2023/06/29**: [SonarSAM](https://github.com/wangsssky/SonarSAM) supports MobileSAM for Image encoder full-finetuing. 
* **2023/06/29**: [Stable Diffusion WebUIv](https://github.com/continue-revolution/sd-webui-segment-anything) supports MobileSAM. 

* **2023/06/28**: [Grounding-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) supports MobileSAM with [Grounded-MobileSAM](https://github.com/IDEA-Research/Grounded-Segment-Anything/tree/main/EfficientSAM). 

* **2023/06/27**: MobileSAM has been featured by [AK](https://twitter.com/_akhaliq?lang=en), see the link [AK's MobileSAM tweet](https://twitter.com/_akhaliq/status/1673585099097636864). Welcome to retweet.
![MobileSAM](assets/model_diagram.jpg?raw=true)

:star: **How is MobileSAM trained?** MobileSAM is trained on a single GPU with 100k datasets (1% of the original images) for less than a day. The training code will be available soon.

:star: **How to Adapt from SAM to MobileSAM?** Since MobileSAM keeps exactly the same pipeline as the original SAM, we inherit pre-processing, post-processing, and all other interfaces from the original SAM. Therefore, by assuming everything is exactly the same except for a smaller image encoder, those who use the original SAM for their projects can **adapt to MobileSAM with almost zero effort**.
 
:star: **MobileSAM performs on par with the original SAM (at least visually)** and keeps exactly the same pipeline as the original SAM except for a change on the image encoder. Specifically, we replace the original heavyweight ViT-H encoder (632M) with a much smaller Tiny-ViT (5M). On a single GPU, MobileSAM runs around 12ms per image: 8ms on the image encoder and 4ms on the mask decoder. 

* The comparison of ViT-based image encoder is summarzed as follows: 

    Image Encoder                                      | Original SAM | MobileSAM 
    :-----------------------------------------:|:---------|:-----:
    Parameters      |  611M   | 5M
    Speed      |  452ms  | 8ms

* Original SAM and MobileSAM have exactly the same prompt-guided mask decoder: 

    Mask Decoder                                      | Original SAM | MobileSAM 
    :-----------------------------------------:|:---------|:-----:
    Parameters      |  3.876M   | 3.876M
    Speed      |  4ms  | 4ms

* The comparison of the whole pipeline is summarized as follows:

    Whole Pipeline (Enc+Dec)                                      | Original SAM | MobileSAM 
    :-----------------------------------------:|:---------|:-----:
    Parameters      |  615M   | 9.66M
    Speed      |  456ms  | 12ms

:star: **Original SAM and MobileSAM with a point as the prompt.**  

<p float="left">
  <img src="assets/mask_point.jpg?raw=true" width="99.1%" />
</p>

:star: **Original SAM and MobileSAM with a box as the prompt.** 
<p float="left">
  <img src="assets/mask_box.jpg?raw=true" width="99.1%" />
</p>

:muscle: **Is MobileSAM faster and smaller than FastSAM? Yes!** 
MobileSAM is around 7 times smaller and around 5 times faster than the concurrent FastSAM. 
The comparison of the whole pipeline is summarzed as follows: 
Whole Pipeline (Enc+Dec)                                      | FastSAM | MobileSAM 
:-----------------------------------------:|:---------|:-----:
Parameters      |  68M   | 9.66M
Speed      |  64ms  |12ms

:muscle: **Does MobileSAM aign better with the original SAM than FastSAM? Yes!** 
FastSAM is suggested to work with multiple points, thus we compare the mIoU with two prompt points (with different pixel distances) and show the resutls as follows. Higher mIoU indicates higher alignment. 
mIoU                                     | FastSAM | MobileSAM 
:-----------------------------------------:|:---------|:-----:
100      |  0.27   | 0.73
200      |  0.33  |0.71
300      |  0.37  |0.74
400      |  0.41  |0.73
500      |  0.41  |0.73








## Installation

The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

### Quick Installation (Recommended)

Install all dependencies from the comprehensive requirements file:

```bash
# Clone the repository
git clone https://github.com/ChaoningZhang/MobileSAM.git
cd MobileSAM

# Install all dependencies
pip install -r requirements.txt

# Install Mobile SAM package
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
```

### Alternative Installation Methods

Install Mobile Segment Anything directly:

```
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
```

or clone the repository locally and install with

```
git clone git@github.com:ChaoningZhang/MobileSAM.git
cd MobileSAM; pip install -e .
```

### Optional Dependencies

The `requirements.txt` includes optional packages that enhance functionality:
- **ONNX Runtime**: For optimized inference (`onnxruntime` or `onnxruntime-gpu`)
- **TensorRT**: For high-performance GPU inference (uncomment in requirements.txt)
- **MLflow**: For experiment tracking and model management
- **Development tools**: Code formatting and linting tools

## Demo

Once installed MobileSAM, you can run demo on your local PC or check out our [HuggingFace Demo](https://huggingface.co/spaces/dhkim2810/MobileSAM).

It requires latest version of [gradio](https://gradio.app).

```
cd app
python app.py
```

## <a name="GettingStarted"></a>Getting Started
The MobileSAM can be loaded in the following ways:

```
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

model_type = "vit_t"
sam_checkpoint = "./weights/mobile_sam.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"

mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam.to(device=device)
mobile_sam.eval()

predictor = SamPredictor(mobile_sam)
predictor.set_image(<your_image>)
masks, _, _ = predictor.predict(<input_prompts>)
```

or generate masks for an entire image:

```
from mobile_sam import SamAutomaticMaskGenerator

mask_generator = SamAutomaticMaskGenerator(mobile_sam)
masks = mask_generator.generate(<your_image>)
```
## <a name="GettingStarted"></a>Getting Started (MobileSAMv2)
Download the model weights from the [checkpoints](https://drive.google.com/file/d/1dE-YAG-1mFCBmao2rHDp0n-PP4eH7SjE/view?usp=sharing).

After downloading the model weights, faster SegEvery with MobileSAMv2 can be simply used as follows:
```
cd MobileSAMv2
bash ./experiments/mobilesamv2.sh
```

## Automatic Mask Generation with Visualization and Structured Output

The `scripts/amg.py` script has been enhanced with comprehensive visualization capabilities and HDF5 structured output that allows users to store and manipulate all segmentation data in a single, efficient file format.

### Output Format Options

#### Traditional Output (Default)
- Individual PNG files or JSON with COCO RLE format
- Optional separate visualization images
- CSV metadata file

#### HDF5 Structured Output (Recommended)
- **Single File**: Everything in one compressed HDF5 file (`.h5`)
- **Complete Data**: Original image, masks, metadata, visualization, and performance metrics
- **Post-Processing**: Filter, analyze, and visualize without re-running segmentation
- **Efficient**: 50-70% smaller than traditional PNG+CSV approach

### Command Line Arguments

#### Core Arguments
- `--input`: Path to input image or folder
- `--output`: Output directory path
- `--model-type`: SAM model type (default: "vit_t")
- `--checkpoint`: Path to SAM checkpoint (default: "weights/mobile_sam.pt")

#### Output Format Arguments
- `--convert-to-rle`: Save as COCO RLE format instead of PNG
- `--hdf5-output`: **Save everything in structured HDF5 format (recommended)**

#### Visualization Arguments
- `--visualize`: Generate visualization images with mask overlays
- `--random-colors`: Use random colors instead of default blue
- `--no-contours`: Disable contour lines in visualizations

#### Performance Tracking
- `--track-performance`: Enable detailed performance metrics
- `--mlflow-tracking`: Log to MLflow (requires mlflow installation)

### Visualization Flag Behavior

⚠️ **Important**: The `--visualize` flag behaves differently depending on the output format:

1. **With `--hdf5-output`**: 
   - Visualization is **automatically included** in the HDF5 file
   - **No separate visualization image files** are saved
   - The `--visualize` flag controls visualization generation but not file output

2. **Without `--hdf5-output` (traditional mode)**:
   - `--visualize` creates separate `{filename}_visualization.png` files
   - Files are saved alongside mask outputs

### Usage Examples

#### HDF5 Structured Output (Recommended)

**Basic HDF5 Output:**
```bash
python scripts/amg.py \
  --input image.jpg \
  --output results/ \
  --hdf5-output
```

**HDF5 with Performance Tracking:**
```bash
python scripts/amg.py \
  --input image.jpg \
  --output results/ \
  --hdf5-output \
  --track-performance \
  --points-per-side 32 \
  --pred-iou-thresh 0.8
```

**Batch Processing with HDF5:**
```bash
python scripts/amg.py \
  --input images_folder/ \
  --output results/ \
  --hdf5-output \
  --track-performance
```

#### Traditional Output with Visualization

**Basic Visualization:**
```bash
python scripts/amg.py \
  --input image.jpg \
  --output results/ \
  --visualize
```

**Visualization with Random Colors:**
```bash
python scripts/amg.py \
  --input image.jpg \
  --output results/ \
  --visualize \
  --random-colors \
  --points-per-side 32
```

### HDF5 Post-Processing and Analysis

#### Installation Requirements
```bash
pip install h5py pandas matplotlib
```

#### Quick Analysis
```python
from scripts.sam_hdf5_utils import load_sam_results

# Load results
results = load_sam_results('results/image.h5')

# Get basic info
info = results.get_info()
print(f"Found {info['num_masks']} masks")

# Filter high-quality masks
quality_masks = results.filter_masks(min_iou=0.8, min_stability=0.9)

# Create visualization
fig = results.visualize_masks(quality_masks)
fig.savefig('quality_analysis.png')
```

#### Advanced Filtering
```python
# Large objects only
large_masks = results.filter_masks(min_area=1000, sort_by='area', top_n=20)

# Specific region analysis
h, w = results.original_image.shape[:2]
center_region = (w//4, h//4, w//2, h//2)
center_masks = results.filter_masks(bbox_filter=center_region, min_area=500)

# Export filtered results
results.export_filtered_masks('filtered_output/', large_masks, format='png')
```

#### Statistics and Comparison
```python
# Get detailed statistics
stats = results.get_statistics()
print(f"Mean area: {stats['area_stats']['mean']:.1f}")
print(f"Mean IoU: {stats['iou_stats']['mean']:.3f}")

# Convert to pandas for analysis
df = results.to_pandas()
df.describe()
```

### Complete Example Workflow

Run comprehensive analysis demonstration:
```bash
python scripts/example_hdf5_usage.py path/to/image.jpg output_directory
```

This will:
1. Run AMG with HDF5 output and performance tracking
2. Load and analyze results with detailed statistics
3. Demonstrate various filtering capabilities
4. Create multiple visualization types
5. Export filtered results in different formats

### HDF5 File Structure
```
your_image.h5
├── metadata/
│   ├── creation_time, image_path, model_type, device
│   ├── amg_* (all AMG parameters)
│   └── performance/ (optional timing metrics)
├── image/
│   └── original (RGB image array with attributes)
├── masks/
│   ├── segmentations (3D array: [mask_id, height, width])
│   └── metadata (structured array with all mask properties)
└── visualization/
    └── image (RGBA visualization with overlays)
```

### Output Files Summary

#### With `--hdf5-output`
- `image.h5`: Single comprehensive file with everything
- No additional image files are created

#### Without `--hdf5-output` (traditional)
- `image/`: Folder with individual mask PNGs + metadata.csv
- `image.json`: COCO RLE format (if `--convert-to-rle`)
- `image_visualization.png`: Overlay image (if `--visualize`)

### Benefits of HDF5 Format

1. **Efficiency**: 50-70% smaller files with compression
2. **Convenience**: Everything in one file
3. **Performance**: Faster loading and selective data access
4. **Rich Analysis**: Built-in filtering, statistics, and visualization tools
5. **Interoperability**: Cross-platform, works with R, MATLAB, etc.
6. **No Re-running**: Post-process without re-running segmentation

### Dependencies

The enhanced features require:
- `h5py`: For HDF5 file operations
- `pandas`: For data analysis and export
- `matplotlib`: For plotting and visualization
- `PIL (Pillow)`: For image manipulation
- `opencv-python`: For contour detection
- `numpy`: For array operations
- `torch`: For GPU acceleration

## ONNX Export and Inference

**MobileSAM** now supports ONNX export and inference with TensorRT acceleration. Export the model with:

### Option 1: Original ONNX Export (Mask Decoder Only)
```bash
python scripts/export_onnx_model.py --checkpoint ./weights/mobile_sam.pt --model-type vit_t --output ./mobile_sam.onnx
```

### Option 2: Separate Image Encoder and Decoder Export (Recommended for AMG)
For automatic mask generation with ONNX models, export the image encoder and mask decoder separately:

```bash
python scripts/export_mobile_sam_onnx.py --checkpoint ./weights/mobile_sam.pt --model-type vit_t --output mobile_sam
```

This creates two files:
- `mobile_sam_encoder.onnx` - Image encoder for computing embeddings
- `mobile_sam_decoder.onnx` - Mask decoder for generating masks from embeddings

### Using ONNX Models with Automatic Mask Generation

Once you have exported ONNX models, you can use them with the enhanced `amg.py` script:

```bash
# Using separate encoder and decoder ONNX models (recommended)
python scripts/amg.py --onnx-model mobile_sam_decoder.onnx --image-encoder-onnx mobile_sam_encoder.onnx --input path/to/images --output output_dir

# Additional options
python scripts/amg.py \
    --onnx-model mobile_sam_decoder.onnx \
    --image-encoder-onnx mobile_sam_encoder.onnx \
    --input path/to/images \
    --output output_dir \
    --device cuda \
    --visualize \
    --track-performance
```

### TensorRT Support

For even faster inference, convert your ONNX models to TensorRT engines:

1. Convert ONNX to TensorRT engine:
```bash
trtexec --onnx=mobile_sam_decoder.onnx --saveEngine=mobile_sam_decoder.trt --fp16
```

2. Use TensorRT engine with AMG:
```bash
python scripts/amg.py --tensorrt-model mobile_sam_decoder.trt --image-encoder-onnx mobile_sam_encoder.onnx --input path/to/images --output output_dir
```

### Requirements

For ONNX support:
```bash
pip install onnxruntime-gpu  # or onnxruntime for CPU only
```

For TensorRT support:
```bash
# Install TensorRT following NVIDIA's official guide
pip install tensorrt
```

We recommend `onnx==1.12.0` and `onnxruntime==1.13.1` which are tested.

Also check the [example notebook](https://github.com/ChaoningZhang/MobileSAM/blob/master/notebooks/onnx_model_example.ipynb) for detailed steps.


## BibTex of our MobileSAM
If you use MobileSAM in your research, please use the following BibTeX entry. :mega: Thank you!

```bibtex
@article{mobile_sam,
  title={Faster Segment Anything: Towards Lightweight SAM for Mobile Applications},
  author={Zhang, Chaoning and Han, Dongshen and Qiao, Yu and Kim, Jung Uk and Bae, Sung-Ho and Lee, Seungkyu and Hong, Choong Seon},
  journal={arXiv preprint arXiv:2306.14289},
  year={2023}
}
```

## Acknowledgement

This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No.RS-2022-00155911, Artificial Intelligence Convergence Innovation Human Resources Development (Kyung Hee University))

<details>
<summary>
<a href="https://github.com/facebookresearch/segment-anything">SAM</a> (Segment Anything) [<b>bib</b>]
</summary>

```bibtex
@article{kirillov2023segany,
  title={Segment Anything}, 
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```
</details>



<details>
<summary>
<a href="https://github.com/microsoft/Cream/tree/main/TinyViT">TinyViT</a> (TinyViT: Fast Pretraining Distillation for Small Vision Transformers) [<b>bib</b>]
</summary>

```bibtex
@InProceedings{tiny_vit,
  title={TinyViT: Fast Pretraining Distillation for Small Vision Transformers},
  author={Wu, Kan and Zhang, Jinnian and Peng, Houwen and Liu, Mengchen and Xiao, Bin and Fu, Jianlong and Yuan, Lu},
  booktitle={European conference on computer vision (ECCV)},
  year={2022}
```
</details>




