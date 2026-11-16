"""First labels of a frame from SAM3-LiteText (docs/labeling.md, research/pool-water-segmentation.md).

The frame is encoded once, then four text prompts: "swimming pool" (the water), "lane rope", "blue stripe" (the floor
lane lines, which "swimming pool" leaves out; they belong to the water) and "person" (swimmers in the water belong to
the pool area too). The masks become shapes (swim/labeling/shapes.py). About 2 s per frame on a MacBook GPU. The first
run downloads the model from Hugging Face (about 2 GB).

License: the image encoder comes from Meta's SAM 3 (SAM License).
"""

import numpy as np

from swim.labeling import shapes
from swim.labeling.labels import Point

MODEL_ID = "vil-uob/sam3-litetext-s0"
PROMPTS = {"water": "swimming pool", "rope": "lane rope", "floor line": "blue stripe", "person": "person"}
MIN_SCORE = 0.4  # instances scoring lower are dropped


class Sam3Labeler:
    name = "sam3-litetext-s0"

    def __init__(self, model_id: str = MODEL_ID):
        # Imported here so the labeling page and the other commands don't load torch / transformers
        import torch
        from transformers import AutoModel, AutoProcessor

        self._torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.texts = {key: self.processor(text=prompt, return_tensors="pt").to(self.device)
                      for key, prompt in PROMPTS.items()}

    def masks(self, frame: np.ndarray) -> dict[str, list[np.ndarray]]:
        """Instance masks for each prompt (keys of PROMPTS), full frame size."""
        from PIL import Image

        h, w = frame.shape[:2]
        image = Image.fromarray(frame[:, :, ::-1])  # BGR -> RGB
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        result = {}
        with self._torch.no_grad():
            vision = self.model.get_vision_features(pixel_values=inputs["pixel_values"])  # once per frame: the slow part
            for key, text in self.texts.items():
                out = self.model(vision_embeds=vision, input_ids=text["input_ids"], attention_mask=text["attention_mask"])
                found = self.processor.post_process_instance_segmentation(
                    out, threshold=MIN_SCORE, mask_threshold=0.5, target_sizes=[(h, w)])[0]
                result[key] = [m.astype(bool) for m in found["masks"].cpu().numpy()]
        return result

    def label(self, frame: np.ndarray) -> tuple[list[list[Point]], list[list[Point]]]:
        """The frame's pool polygons and rope polygons (each rope's band: 4 corners)."""
        m = self.masks(frame)
        water = np.zeros(frame.shape[:2], bool)
        for mask in m["water"]:
            water |= mask
        water_polygons = shapes.pool_polygons(water, persons=m["person"], parts=m["rope"] + m["floor line"])
        return water_polygons, [rope.outline() for rope in shapes.ropes_from_masks(m["rope"])]
