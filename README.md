# TBG_Enhanced Tiled Upscaler & Refiner FLUX PRO


Welcome to **TBG Enhanced Tiled Upscaler and Refiner Pro!**
We at **TBG Think. Build. Generate. AI Enritchment** are
excited to make our TBG Enhanced Tiled Upscaler and Refiner Pro available to you
for free non-commercial use. We believe in empowering creators and innovators,
which is why anything you create or generate using our software is completely
yours to use however you wish—including for commercial purposes!

What This Means for You:
Your creations are truly yours: Any images, outputs, or content you generate with
our software can be used freely for personal projects, commercial ventures,
selling, publishing, or any other purpose you have in mind.
Free to use for personal and non-commercial purposes: You can use our Community
Edition without any cost for personal projects, research, education, or hobby use.

Important Limitations to Remember:
While we want you to have maximum freedom with your creations, there are some
important boundaries:
- The software itself (not your outputs) is for non-commercial use only
- You cannot modify, sell, or redistribute the software without our permission
- The Pro Version requires API access authorization
- This is alpha software, so use it at your own risk

Check our license agreement.

# Alpha Testing of PRO Features Now Available

We’re excited to announce that alpha testing of the PRO version is now live
for our Patreon supporters and free community members!

Get early access by joining us at:
https://www.patreon.com/TB_LAAR

You must be a TB_LAAR Patreon supporter or free member to get an API key which you can get here:
https://api.ylab.es/login.php 


**TBG Enhanced Tiled Upscaler and Refiner Pro** is an advanced, modular enhancement suite for **tiled image generation and refinement** in ComfyUI. It introduces neuro generative tile fusion, interactive tile-based editing, and multi-path processing pipelines designed for extreme resolution workflows ut to but not limitet to 100MP, high fidelity, and adaptive post-generation control.

## 1. New Fusion Techniques

TBG Enhanced Tiled Upscaler and Refiner Pro introduces tree next-generation tile fusion processes that go far beyond traditional blending:

### Smart Merge CE
Choose between adaptive blending strategies compatible with common upscalers like ESRGAN, SwinIR, and others. It intelligently handles tile overlaps, avoiding ghosting and seams.

### PRO Tile Diffusion  
A novel approach where each tile is generated while considering surrounding tile information — perfect for mid-denoise levels. This enables:
- Context-aware detail consistency  
- Smooth transitions across tile borders  
- Better handling of textures and patterns  

### PRO Neuro-Generative Tile Fusion (NGTF)  
An advanced generative system that remembers newly generated surroundings and adapts subsequent sampling steps accordingly. This makes high-denoise tile refinement possible while maintaining:
- Global consistency  
- Sharp, coherent details  
- Memory of contextual relationships across tiles  

## 2. Design Suite

TBG_Enhanced is not only about quality — it's about flexible, editable workflows over time:

### One-Tile Preview & Smart Presets  
Quickly generate single-tile previews to fine-tune the right settings before running the full job. Presets adapt intelligently to image dimensions and desired resolution.

### Post-Sampling Tile Correction  
You can resample only the tiles you don’t like — no need to regenerate the full image. This allows:
- Precision editing  
- Fixing small errors without full reprocessing  

### Resume Tile Refinement  
Modify or refine individual tiles days later while keeping the original input image and final result. The system supports:
- Saved tile maps and settings  
- Re-injection of noise and conditioning  
- Fully restorable editing workflows  

## 3. Pipeline Structure

TBG_Enhanced is powered by a flexible pipeline architecture built around tile-aware processing paths:

### TBG Tile Prompter Pipe  
Access prompt and denoise settings per tile. Enables:
- Per-region storytelling  
- Adaptive text-to-image behavior  
- Denoising strength by tile  

### TBG Tile Enrichment Pipe  
Control multiple sampling and model-level features:
- Model-side: Use DemonTools for deep model manipulation  
- Sampler-side: Inject custom noise at specific steps, or apply sigma curves to selected steps  
- Sampler-internal: Enable per-step sampler-side noise injection  
- Built-in noise reduction  
- Optional tile up/downscaling during sampling  

### ControlNet Pipe  
Tiled generation now supports unlimited ControlNet inputs per tile, unlocking:
- High-resolution conditioning  
- Fine-grained control for large images  
- Targeted structure, depth, edge, pose, or segmentation maps for each region  


## Getting Started

1. Clone into your ComfyUI custom nodes directory:
   
   https://github.com/Ltamann/ComfyUI-TBG-ETUR/tree/main
   

2. Restart ComfyUI. Nodes will appear under the `TBG_Enhanced` category.

3. (Optional) Place the `flux_pro_refiner.safetensors` model into your `models/` folder if using FLUX PRO.

## Status and Roadmap

- Fusion Techniques: Smart Merge, Tile Diffusion, NGTF  
- Post-editing Tools: One-tile preview, correction, resume-refine  
- Pipelines: Prompt / Enrichment / ControlNet Pipes  
- Upcoming: Full integration with custom samplers and DreamEdit compatibility  



Instructions for Installing TBG_upscaler_Alfa_1.02p for ComfyUI
--------------------------------------------------------------

1. Copy the folder 'TBG_upscaler_Alfa_1.02p' into your ComfyUI custom_nodes directory:
   ..\ComfyUI\custom_nodes\

2. Open your ComfyUI Python environment and run the following command:
   ..\ComfyUI\custom_nodes\TBG_upscaler_Alfa_1.02p\python -m install.py

3. Get early access to PRO feachers by joining us at:
   https://www.patreon.com/TB_LAAR

   You must be a TB_LAAR Patreon supporter or free member to get an API key which you can get here:
   https://api.ylab.es/login.php 


   You can use your API key in two ways:
   - Paste it directly into the TBG Tiler node
   - OR set an environment variable named: TBG_ETUR_API_KEY
      This will install the required packages from requirements.txt
      and download the model file: depth_anything_v2_vitl.pth

--------------------------------------------------------------

Alternative Manual Installation:

1. Download the model file manually from:
   https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/cbbb86a30ce19b5684b7a05155dc7e6cbc7685b9/depth_anything_v2_vitl.pth

2. Copy the downloaded file to the following folder:
   ..\ComfyUI\custom_nodes\TBG_upscaler_Alfa_1.02p\py\vendor\comfyui_controlnet_aux\ckpts\depth-anything\Depth-Anything-V2-Large

3. Install the required Python packages by running:
   ..\ComfyUI\custom_nodes\TBG_upscaler_Alfa_1.02p\python -m pip install -r requirements.txt

--------------------------------------------------------------

Important:
To use PRO features, you must stay connected to the internet.
If you're offline, you can still use all the Community Edition features for free.

Recommended Workflow:
Do your setup and testing with PRO features turned OFF,
and only enable them for the final steps of your workflow.

Thank you for your support and happy tiling!
— TBG 

