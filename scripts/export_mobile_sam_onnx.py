#!/usr/bin/env python3
"""
Export Mobile SAM models to ONNX format for use with amg.py

This script exports both the image encoder and mask decoder portions of Mobile SAM
to separate ONNX files, which can then be used with the modified amg.py script.

Usage:
    python scripts/export_mobile_sam_onnx.py --checkpoint weights/mobile_sam.pt --output mobile_sam
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import os
from typing import Tuple

from mobile_sam import sam_model_registry
from mobile_sam.utils.onnx import SamOnnxModel


class MobileSamImageEncoder(nn.Module):
    """Wrapper for SAM image encoder to enable ONNX export."""
    
    def __init__(self, sam_model):
        super().__init__()
        self.image_encoder = sam_model.image_encoder
        self.img_size = sam_model.image_encoder.img_size
        self.pixel_mean = sam_model.pixel_mean
        self.pixel_std = sam_model.pixel_std
        
    def forward(self, x):
        """Forward pass through image encoder only."""
        # x is expected to be preprocessed (normalized) image
        return self.image_encoder(x)


def export_image_encoder(model, output_path: str, opset: int = 16):
    """Export the image encoder to ONNX."""
    encoder_model = MobileSamImageEncoder(model)
    encoder_model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 1024, 1024)  # Mobile SAM uses 1024x1024 input
    
    # Export to ONNX
    print(f"Exporting image encoder to {output_path}")
    torch.onnx.export(
        encoder_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['embeddings'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'embeddings': {0: 'batch_size'}
        }
    )


def export_mask_decoder(model, output_path: str, opset: int = 16, return_single_mask: bool = False):
    """Export the mask decoder to ONNX."""
    onnx_model = SamOnnxModel(
        model=model,
        return_single_mask=return_single_mask,
        use_stability_score=False,
        return_extra_metrics=False,
    )
    onnx_model.eval()
    
    # Create dummy inputs matching the expected format
    embed_dim = model.prompt_encoder.embed_dim
    embed_size = model.prompt_encoder.image_embedding_size
    mask_input_size = [4 * x for x in embed_size]
    
    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
        "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input": torch.tensor([1], dtype=torch.float),
        "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
    }
    
    # Export to ONNX
    print(f"Exporting mask decoder to {output_path}")
    torch.onnx.export(
        onnx_model,
        tuple(dummy_inputs.values()),
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=list(dummy_inputs.keys()),
        output_names=["masks", "iou_predictions", "low_res_masks"],
        dynamic_axes={
            "point_coords": {1: "num_points"},
            "point_labels": {1: "num_points"},
        }
    )


def main():
    parser = argparse.ArgumentParser(description="Export Mobile SAM to ONNX format")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Mobile SAM checkpoint")
    parser.add_argument("--model-type", type=str, default="vit_t", help="Model type (vit_t, vit_b, etc.)")
    parser.add_argument("--output", type=str, required=True, help="Output prefix for ONNX files")
    parser.add_argument("--opset", type=int, default=16, help="ONNX opset version")
    parser.add_argument("--return-single-mask", action="store_true", help="Export decoder with single mask output")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading {args.model_type} model from {args.checkpoint}")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.eval()
    
    # Export image encoder
    encoder_path = f"{args.output}_encoder.onnx"
    export_image_encoder(sam, encoder_path, args.opset)
    
    # Export mask decoder  
    decoder_path = f"{args.output}_decoder.onnx"
    export_mask_decoder(sam, decoder_path, args.opset, args.return_single_mask)
    
    print(f"\nExport complete!")
    print(f"Image encoder: {encoder_path}")
    print(f"Mask decoder: {decoder_path}")
    print(f"\nTo use with amg.py:")
    print(f"python scripts/amg.py --onnx-model {decoder_path} --image-encoder-onnx {encoder_path} --input <input> --output <output>")


if __name__ == "__main__":
    main() 