import numpy as np
import torch
import cv2
import logging
from mobile_sam import sam_model_registry, SamPredictor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SegmentationEngine:
    def __init__(self, checkpoint_path=None, model_type="vit_b", device=None, model_instance=None):
        """
        Initialize the SAM model.
        Args:
            checkpoint_path: Path to weights (if loading new).
            model_type: SAM architecture type.
            device: 'cuda' or 'cpu'.
            model_instance: Pre-loaded sam_model_registry instance (optional).
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        if model_instance is not None:
             self.sam = model_instance
        elif checkpoint_path:
             logger.info(f"Loading SAM model ({model_type}) on {self.device}...")
             self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
             self.sam.to(device=self.device)
        else:
             raise ValueError("Either checkpoint_path or model_instance must be provided.")

        self.predictor = SamPredictor(self.sam)
        self.is_image_set = False

    def set_image(self, image_rgb):
        """
        Process the image and compute embeddings.
        Args:
            image_rgb: NumPy array (H, W, 3) in RGB format.
        """
        logger.info("Computing image embeddings...")
        self.predictor.set_image(image_rgb)
        self.is_image_set = True
        logger.info("Embeddings computed.")
        self.image_rgb = image_rgb # Store for cleanup logic

    def generate_mask(self, point_coords=None, point_labels=None, box_coords=None, level=None, cleanup=True):
        """
        Generate a mask for a given point or box.
        Args:
            point_coords: List of [x, y] or NumPy array.
            point_labels: List of labels (1 for foreground, 0 for background).
            box_coords: [x1, y1, x2, y2]
            level: int (0, 1, 2) or None. 
                   0=Fine Details, 1=Sub-segment, 2=Whole Object. 
                   If None, auto-selects highest score.
            cleanup: bool. If True, removes disconnected components to prevent leaks.
        """
        if not self.is_image_set:
            raise RuntimeError("Image not set. Call set_image() first.")

        # Prepare inputs
        sam_point_coords = None
        sam_point_labels = None
        sam_box = None

        if point_coords is not None:
            sam_point_coords = np.array(point_coords)
            if point_labels is None:
                sam_point_labels = np.array([1] * len(point_coords))
            else:
                sam_point_labels = np.array(point_labels)
        
        if box_coords is not None:
            sam_box = np.array(box_coords)

        with torch.inference_mode():
            masks, scores, logits = self.predictor.predict(
                point_coords=sam_point_coords,
                point_labels=sam_point_labels,
                box=sam_box,
                multimask_output=True # Generate multiple masks and choose best
            )

        # Select best mask
        if level is not None and 0 <= level < 3:
            # User forced a specific level
            if level == 1:
                best_mask = masks[0]
            elif level == 0:
                # --- PHOTOSHOP BOX MODE ---
                # If we have a box, we ALWAYS want the largest/whole object (Index 2)
                # This ensures "entire selection object" coverage.
                if box_coords is not None:
                    best_mask = masks[2] if scores[2] > 0.4 else masks[1]
                else:
                    best_mask = masks[0]
            else:
                best_mask = masks[level]
        else:
            # Heuristic: Favor 'Whole' (Index 2) for box prompts to match user's "entire object" request
            if box_coords is not None:
                best_mask = masks[2] if scores[2] > 0.4 else masks[1]
            else:
                if scores[1] > 0.70: 
                    best_mask = masks[1]
                else:
                    best_idx = np.argmax(scores)
                    best_mask = masks[best_idx]
        
        if cleanup:
            h, w = best_mask.shape
            mask_uint8 = (best_mask * 255).astype(np.uint8)
            
            # Use a reference point for connectivity filtering
            ref_x, ref_y = None, None
            if point_coords is not None and len(point_coords) > 0:
                pos_indices = np.where(sam_point_labels == 1)[0]
                if len(pos_indices) > 0:
                    idx = pos_indices[-1]
                    ref_x, ref_y = int(sam_point_coords[idx][0]), int(sam_point_coords[idx][1])
            elif box_coords is not None:
                ref_x = int((box_coords[0] + box_coords[2]) / 2)
                ref_y = int((box_coords[1] + box_coords[3]) / 2)

            if ref_x is not None:
                # --- ADAPTIVE FILTERING ---
                # For BOX prompts, we trust SAM much more and disable the strict color guard
                # to allow painting "entire selection objects" even with shadows/holes.
                if box_coords is not None:
                    valid_mask = np.ones((h, w), dtype=np.uint8)
                    edge_barrier = np.ones((h, w), dtype=np.uint8)
                    mask_refined = mask_uint8
                else:
                    # Point click still needs safety to prevent bleeding
                    y1, y2 = max(0, ref_y-1), min(h, ref_y+2)
                    x1, x2 = max(0, ref_x-1), min(w, ref_x+2)
                    seed_patch = self.image_rgb[y1:y2, x1:x2]
                    seed_color = np.mean(seed_patch, axis=(0, 1))
                    
                    std_dev = np.std(seed_color)
                    is_grayscale_seed = std_dev < 10.0
                    
                    img_u16 = self.image_rgb.astype(np.uint16)
                    intensity_dist = np.abs(np.mean(img_u16, axis=2) - np.mean(seed_color))
                    
                    if level == 0:
                        valid_mask = (intensity_dist < 150).astype(np.uint8) # Relaxed
                    else:
                        valid_mask = (intensity_dist < 100).astype(np.uint8)

                    # Edge detection
                    k_size = (11, 11) if level == 0 else (5, 5)
                    edge_gray = cv2.GaussianBlur(cv2.cvtColor(self.image_rgb, cv2.COLOR_RGB2GRAY), k_size, 0)
                    edges = cv2.Laplacian(edge_gray, cv2.CV_16S, ksize=3)
                    abs_edges = cv2.convertScaleAbs(edges)
                    _, edge_barrier = cv2.threshold(abs_edges, 50, 255, cv2.THRESH_BINARY_INV)
                    edge_barrier = (edge_barrier / 255).astype(np.uint8)
                    cv2.circle(edge_barrier, (ref_x, ref_y), 5, 1, -1)
                    
                    mask_refined = (mask_uint8 & valid_mask & edge_barrier)

                # --- AGGRESSIVE HOLE FILLING ---
                # For Walls and Boxes, we fill all internal holes to ensure "entire selection" coverage.
                # This is critical for perforated walls (like in the user's example photo).
                if level == 0 or box_coords is not None:
                    # Closing operation to bridge architectural gaps (Perforated holes, Textures)
                    # We use a very large kernel to bridge separated holes into one solid wall.
                    k_size = (35, 35) if box_coords is not None else (15, 15)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k_size)
                    mask_refined = cv2.morphologyEx(mask_refined, cv2.MORPH_CLOSE, kernel)
                    
                    # Fill ALL internal holes immediately
                    # RECOMP: Find all contours and fill everything that isn't the outer background
                    contours, _ = cv2.findContours(mask_refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    mask_solid = np.zeros_like(mask_refined)
                    cv2.drawContours(mask_solid, contours, -1, 1, thickness=cv2.FILLED)
                    mask_refined = mask_solid

                if np.sum(mask_refined) > 50:
                    mask_uint8 = mask_refined
            
            # Connectivity filtering and Object Recovery
            if ref_x is not None:
                num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
                if num_labels > 1:
                    if box_coords is not None:
                        # --- BOX MODE: MULTI-COMPONENT RECOVERY ---
                        # In Photoshop style, if you box an object with holes (like a perforated wall),
                        # we want to keep ALL pieces of that object that are inside the box.
                        recovered_mask = np.zeros_like(best_mask)
                        bx1, by1, bx2, by2 = box_coords
                        
                        for i in range(1, num_labels):
                            cx, cy = centroids[i]
                            # If centroid is inside box, or it's the largest component
                            if (bx1 < cx < bx2 and by1 < cy < by2) or stats[i, cv2.CC_STAT_AREA] > (h*w*0.05):
                                recovered_mask |= (labels_im == i)
                        
                        if np.any(recovered_mask):
                            best_mask = recovered_mask
                    else:
                        # Point Click Mode: Keep only the target component
                        ref_x = max(0, min(ref_x, w - 1))
                        ref_y = max(0, min(ref_y, h - 1))
                        target_label = labels_im[ref_y, ref_x]
                        if target_label != 0:
                            best_mask = (labels_im == target_label)
                        else:
                            max_area = 0
                            max_label = 1
                            for i in range(1, num_labels):
                                if stats[i, cv2.CC_STAT_AREA] > max_area:
                                    max_area = stats[i, cv2.CC_STAT_AREA]
                                    max_label = i
                            best_mask = (labels_im == max_label)
        
        return best_mask

