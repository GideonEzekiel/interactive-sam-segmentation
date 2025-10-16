import torch
from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np
from PIL import Image
import gradio as gr
import os
from datetime import datetime

# --- 1. Global Initialization ---
# NOTE: The SAM checkpoint file ("sam_vit_b.pth") must be available in the same directory.
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b.pth")
sam.to(device)

predictor = SamPredictor(sam)

# --- Global State Variables ---
image = None               # Stores the original uploaded image (BGR)
clicks = []                # Stores temporary clicks (coords, label) for *current* segment (used temporarily in segment_with_click)
last_mask = None           # Stores the last completed mask (for saving)
segment_history = []       # Stores completed segments as (mask, hex_color) tuples

# --- 2. Core Segmentation Functions ---

def set_image(uploaded_img):
  """Initializes the tool when a new image is uploaded."""
  global image, clicks, segment_history
  image = cv2.cvtColor(np.array(uploaded_img), cv2.COLOR_RGB2BGR)
  predictor.set_image(image)
  clicks=[]
  segment_history = [] # Clear history for a new image
  # NOTE: The predictor must be set again to reset its internal state
  return uploaded_img

def segment_with_click(x, y, click_type="positive"):
  """Performs the SAM prediction based on a single click."""
  # We only use one click per segment in the multi-segment workflow
  label = 1 if click_type == "positive" else 0 
  
  # Predict based ONLY on the current click for a new object
  input_points = np.array([[x, y]])
  input_labels = np.array([label])

  masks, scores, _ = predictor.predict(
      point_coords = input_points,
      point_labels = input_labels,
      multimask_output=True,
  )

  # Select the highest scoring mask
  mask = masks[np.argmax(scores)]
  return mask.astype(np.uint8)

def redraw_overlays():
    """Iterates through segment_history and redraws all masks onto a clean image copy."""
    global image, segment_history
    if image is None:
        return None

    # Start with a clean copy of the original image
    current_overlay = image.copy()

    for mask, hex_color in segment_history:
        # Convert hex to BGR
        hex_color = hex_color.lstrip("#")
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        bgr = (rgb[2], rgb[1], rgb[0])

        # Create a color layer for *this* specific mask
        mask_color = np.zeros_like(image)
        mask_color[mask == 1] = bgr 

        # Apply the overlay with 0.4 opacity
        current_overlay = cv2.addWeighted(current_overlay, 1.0, mask_color, 0.4, 0)

    # Convert from BGR (OpenCV) to RGB (Gradio/PIL) for display
    return cv2.cvtColor(current_overlay, cv2.COLOR_BGR2RGB)


# --- 3. Event Handlers and Utility Functions ---

def click_or_box_handler(img, evt:gr.SelectData, color):
  """Handles the click event: segments, updates history, and redraws."""
  global image, last_mask, segment_history
  if image is None or evt is None or evt.index is None:
    return None

  x, y = evt.index
  print(f"clicked at {x}, {y}")

  # 1. Get the new mask 
  mask = segment_with_click(x, y) 

  if mask is None:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  
  # 2. Finalize the segment: Store the mask and color
  segment_history.append((mask, color))
  last_mask = mask.astype(np.uint8)*255 # Store the last segment mask for saving

  # 3. Redraw all overlays
  result = redraw_overlays() 
  return result

def clear_clicks():
  """Resets the display and clears all segment history."""
  global clicks, image, segment_history
  clicks=[]
  segment_history = [] 
  if image is not None:
    predictor.set_image(image) # Reset predictor
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  return None

def save_mask():
  """Saves the binary mask of the last completed segment."""
  global last_mask
  if last_mask is None:
    print("Nothing to save")
    return None
  
  os.makedirs("masks", exist_ok=True)
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  filename = f"masks/mask_{timestamp}.png"
  
  Image.fromarray(last_mask).save(filename) 

  return filename

# --- 4. Gradio Interface ---

with gr.Blocks() as demo:
  gr.Markdown("# Interactive Multi-Color Segmentation with SAM")
  gr.Markdown("Click an area to segment an object. Choose a new color and click again to segment a *different* object.")
  
  with gr.Row():
    img_input = gr.Image(type="pil",
                         label="Upload and Click")
    img_output = gr.Image(label="Segmented Output")

    with gr.Column():
      color_picker = gr.ColorPicker(label="Choose mask color", value="#ff0000")
      # NOTE: For deployment on Hugging Face Spaces, you may want to remove `share=True`
      save_btn = gr.Button("Save Mask (Last Segment Only)")
      clear_btn = gr.Button("RESET ALL SEGMENTS")
      download_file=gr.File(label="Download Saved Mask")

  # Connect event handlers
  img_input.change(fn=set_image, inputs=img_input, outputs=img_output)
  img_input.select(fn=click_or_box_handler, inputs=[img_input, color_picker], outputs=img_output)

  clear_btn.click(fn=clear_clicks, outputs=img_output)
  save_btn.click(fn=save_mask, outputs=download_file)

# Launch the app when run directly
if __name__ == "__main__":
    # Remove share=True if deploying to Hugging Face Spaces
    demo.launch(share=True)
    