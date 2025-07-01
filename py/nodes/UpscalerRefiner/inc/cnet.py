import comfy.model_management as model_management
import nodes
from ..inc.sigmas import process_image_to_tiles
from ....vendor.comfyui_controlnet_aux.src.custom_controlnet_aux.canny.canny import CannyDetector
from ....vendor.comfyui_controlnet_aux.src.custom_controlnet_aux.depth_anything_v2.da2tgb import DepthAnythingV2Detector
from ....vendor.comfyui_controlnet_aux.utils import common_annotator_call


def apply_controlnets_from_pipe(self, cnetpipe, positive, negative, full_image, tile_image, vae):

    controlnet_node = nodes.ControlNetApplyAdvanced()
    for control in cnetpipe:
        controlnet_model = control["controlnet"]
        strength = control["strength"]
        start = control["start"]
        end = control["end"]
        preprocessor = control["preprocessor"]
        canny_high_threshold = control["canny_high_threshold"]
        canny_low_threshold = control["canny_low_threshold"]
        noise_image = control["noise_image"]
        # set image for CNET
        cnet_image = tile_image
        if noise_image is not None:
            grid_images = process_image_to_tiles(self,noise_image)
            cnet_image = grid_images[self.KSAMPLER.latent_index]
            if isinstance(cnet_image, tuple):
                cnet_image = np.array(cnet_image)

        strength = strength*self.KSAMPLER.cnet_multiply 
        # Preprocessero
        if preprocessor=="Canny":
            cnet_image = Canny().detect_edge(cnet_image, canny_low_threshold/100, canny_high_threshold/100)[0]
        if preprocessor=="DepthAnythingV2":  
            model = DepthAnythingV2Detector.from_pretrained(filename="depth_anything_v2_vitl.pth").to(model_management.get_torch_device())
            cnet_image = common_annotator_call(model, cnet_image, resolution=1024, max_depth=1)
            del model    
        if preprocessor=="Canny Edge":
            cnet_image = common_annotator_call(CannyDetector(), cnet_image, canny_low_threshold=canny_low_threshold, canny_high_threshold=canny_high_threshold, resolution=1024)
        positive, negative = controlnet_node.apply_controlnet(
            positive, negative, controlnet_model, cnet_image, strength, start, end, vae
        )
    return positive, negative