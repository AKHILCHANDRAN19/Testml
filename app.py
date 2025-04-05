import os
import cv2
import numpy as np
from g4f.client import Client
import pollinations

##############################
# Directory Setup
##############################
DOWNLOADS_DIR = '/storage/emulated/0/Download'
IMAGES_FOLDER = os.path.join(DOWNLOADS_DIR, "GeneratedImages")
if not os.path.exists(IMAGES_FOLDER):
    os.makedirs(IMAGES_FOLDER)
# Folders for temporary clip videos and transitions
CLIPS_FOLDER = os.path.join(IMAGES_FOLDER, "ClipVideos")
if not os.path.exists(CLIPS_FOLDER):
    os.makedirs(CLIPS_FOLDER)
TRANS_FOLDER = os.path.join(IMAGES_FOLDER, "Transitions")
if not os.path.exists(TRANS_FOLDER):
    os.makedirs(TRANS_FOLDER)
# Final video output path
FINAL_VIDEO_PATH = os.path.join(DOWNLOADS_DIR, "final_video.mp4")

##############################
# Pollinations Image Generation
##############################
def generate_image_prompt(full_paragraph, sentence):
    """
    Uses g4f to generate an image-generation prompt for the given sentence,
    taking the full paragraph as context.
    """
    client = Client()
    message = (
        f"Given the full paragraph: \"{full_paragraph}\", "
        f"generate an image generation prompt for the sentence: \"{sentence}\". "
        "Ensure the prompt reflects the context of the full paragraph and is in English."
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": message}],
        web_search=False
    )
    return response.choices[0].message.content

def generate_images(prompt, sentence_number, num_images=2):
    """
    Uses pollinations to generate images with settings:
      - Model: flux_3d
      - Enhance: True
      - Dimensions: 720x1280 (9:16)
    Saves the images in the GeneratedImages folder with names:
      "sentence {sentence_number} image {i}.png"
    Returns a list of file paths.
    """
    image_model = pollinations.Image(
        model="flux_3d",
        seed="random",
        width=720,
        height=1280,
        enhance=True,
        nologo=True,
        private=True,
        safe=False,
        referrer="pollinations.py"
    )
    
    file_paths = []
    for i in range(1, num_images + 1):
        image_obj = image_model(prompt=prompt)
        file_name = os.path.join(IMAGES_FOLDER, f"sentence {sentence_number} image {i}.png")
        image_obj.save(file=file_name)
        print(f"Saved generated image: {file_name}")
        file_paths.append(file_name)
    return file_paths

##############################
# Effect Functions (each clip: 3.5s = 105 frames at 30 FPS)
##############################
FRAMES_PER_CLIP = 105
FPS = 30
FRAME_SIZE = (720, 1280)

def effect_zoom_in_small(img, frames=FRAMES_PER_CLIP):
    """Apply a small zoom in effect: scale from 1.0 to 0.93."""
    h, w = img.shape[:2]
    for i in range(frames):
        factor = 1.0 - (i / (frames - 1)) * 0.07  # from 1.0 to 0.93
        new_w = int(w * factor)
        new_h = int(h * factor)
        x = (w - new_w) // 2
        y = (h - new_h) // 2
        cropped = img[y:y+new_h, x:x+new_w]
        frame = cv2.resize(cropped, (w, h))
        yield frame

def effect_dolly_zoom_small(img, frames=FRAMES_PER_CLIP):
    """Apply a small dolly zoom effect: scale from 1.0 to 0.94 with horizontal offset up to 15 pixels."""
    h, w = img.shape[:2]
    max_offset = 15
    for i in range(frames):
        scale = 1.0 - (i / (frames - 1)) * 0.06  # from 1.0 to 0.94
        new_w = int(w * scale)
        new_h = int(h * scale)
        offset = int((i / (frames - 1)) * max_offset)
        x = (w - new_w) // 2 + offset
        y = (h - new_h) // 2
        x = max(0, min(x, w - new_w))
        y = max(0, min(y, h - new_h))
        cropped = img[y:y+new_h, x:x+new_w]
        frame = cv2.resize(cropped, (w, h))
        yield frame

def effect_zoom_out_small(img, frames=FRAMES_PER_CLIP):
    """Apply a small zoom out effect: scale from 0.93 to 1.0."""
    h, w = img.shape[:2]
    for i in range(frames):
        factor = 0.93 + (i / (frames - 1)) * 0.07  # from 0.93 to 1.0
        new_w = int(w * factor)
        new_h = int(h * factor)
        x = (w - new_w) // 2
        y = (h - new_h) // 2
        cropped = img[y:y+new_h, x:x+new_w]
        frame = cv2.resize(cropped, (w, h))
        yield frame

def effect_pull_out_shake(img, frames=FRAMES_PER_CLIP):
    """Apply a pull out shake effect: scale from 0.93 to 1.0 while adding random shake (offset up to 8 pixels)."""
    h, w = img.shape[:2]
    for i in range(frames):
        factor = 0.93 + (i / (frames - 1)) * 0.07  # from 0.93 to 1.0
        new_w = int(w * factor)
        new_h = int(h * factor)
        x = (w - new_w) // 2
        y = (h - new_h) // 2
        offset_x = np.random.randint(-8, 9)
        offset_y = np.random.randint(-8, 9)
        x = max(0, min(x + offset_x, w - new_w))
        y = max(0, min(y + offset_y, h - new_h))
        cropped = img[y:y+new_h, x:x+new_w]
        frame = cv2.resize(cropped, (w, h))
        yield frame

# List of effect functions in cyclic order
effect_functions = [
    effect_zoom_in_small,
    effect_dolly_zoom_small,
    effect_zoom_out_small,
    effect_pull_out_shake
]

##############################
# Transition Functions (each transition: 1s = 30 frames)
##############################
TRANSITION_FRAMES = 30

def transition_fire(img1, img2, frames=TRANSITION_FRAMES):
    """
    Fire transition: applies a fiery effect blending img1 to img2.
    This function simulates a fire-like transition by blending the images,
    applying a heatmap (COLORMAP_HOT), and adding some noise.
    """
    output_frames = []
    for i in range(frames):
        alpha = i / (frames - 1)
        blend = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        # Apply a hot colormap for fire effect
        fire_effect = cv2.applyColorMap(blend, cv2.COLORMAP_HOT)
        # Add random noise to simulate flickering
        noise = np.random.randint(0, 50, img1.shape, dtype=np.uint8)
        frame = cv2.addWeighted(fire_effect, 0.8, noise, 0.2, 0)
        output_frames.append(frame)
    return output_frames

def transition_white(img1, img2, frames=TRANSITION_FRAMES):
    """White transition: fade from img1 to white then from white to img2."""
    output_frames = []
    h, w = img1.shape[:2]
    white = np.ones((h, w, 3), dtype=np.uint8) * 255
    half = frames // 2
    for i in range(half):
        alpha = i / half
        frame = cv2.addWeighted(img1, 1 - alpha, white, alpha, 0)
        output_frames.append(frame)
    for i in range(frames - half):
        alpha = i / (frames - half)
        frame = cv2.addWeighted(white, 1 - alpha, img2, alpha, 0)
        output_frames.append(frame)
    return output_frames

def transition_glitch(img1, img2, frames=TRANSITION_FRAMES):
    """Glitch transition: applies a stripe-based glitch effect between img1 and img2."""
    output_frames = []
    h, w = img1.shape[:2]
    num_stripes = 10
    stripe_height = h // num_stripes
    for _ in range(frames):
        frame = np.zeros_like(img1)
        for stripe in range(num_stripes):
            y1 = stripe * stripe_height
            y2 = y1 + stripe_height if stripe < num_stripes - 1 else h
            offset = np.random.randint(-10, 11)
            blend_factor = np.random.rand()
            stripe1 = img1[y1:y2, :]
            stripe2 = img2[y1:y2, :]
            M = np.float32([[1, 0, offset], [0, 1, 0]])
            shifted_stripe = cv2.warpAffine(stripe1, M, (w, y2 - y1))
            blended_stripe = cv2.addWeighted(shifted_stripe, blend_factor, stripe2, 1 - blend_factor, 0)
            frame[y1:y2, :] = blended_stripe
        output_frames.append(frame)
    return output_frames

# Transition functions list (cyclic order): now fire, white, glitch
transition_functions = [
    transition_fire,
    transition_white,
    transition_glitch
]

##############################
# Functions to Save Clips and Transitions
##############################
def save_effect_clip(image_path, effect_func, clip_index):
    """
    Reads an image from image_path, applies the given effect function (yielding frames),
    and saves the frames as a video file in CLIPS_FOLDER.
    Returns the saved clip file path.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None
    img = cv2.resize(img, FRAME_SIZE)
    clip_filename = os.path.join(CLIPS_FOLDER, f"clip_{clip_index}.mp4")
    writer = cv2.VideoWriter(clip_filename, cv2.VideoWriter_fourcc(*'mp4v'), FPS, FRAME_SIZE)
    for frame in effect_func(img, FRAMES_PER_CLIP):
        writer.write(frame)
    writer.release()
    print(f"Saved effect clip: {clip_filename}")
    return clip_filename

def save_transition_clip(last_frame, first_frame, trans_func, trans_index):
    """
    Uses the given transition function (yielding frames) to generate transition frames
    between last_frame and first_frame, and saves them as a video file in TRANS_FOLDER.
    Returns the saved transition file path.
    """
    trans_filename = os.path.join(TRANS_FOLDER, f"transition_{trans_index}.mp4")
    writer = cv2.VideoWriter(trans_filename, cv2.VideoWriter_fourcc(*'mp4v'), FPS, FRAME_SIZE)
    for frame in trans_func(last_frame, first_frame, TRANSITION_FRAMES):
        writer.write(frame)
    writer.release()
    print(f"Saved transition clip: {trans_filename}")
    return trans_filename

def get_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def get_last_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    last_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        last_frame = frame
    cap.release()
    return last_frame

def merge_video_files(video_files, output_path):
    """
    Merges the video files in the list video_files (in order) into one final video.
    """
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), FPS, FRAME_SIZE)
    for file in video_files:
        cap = cv2.VideoCapture(file)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
        cap.release()
    writer.release()
    print(f"Final video saved to: {output_path}")

##############################
# Main Execution
##############################
def main():
    # Generate images from paragraph input.
    paragraph = input("Enter your paragraph: ").strip()
    sentences = [s.strip() for s in paragraph.split('.') if s.strip()]
    all_image_paths = []
    for idx, sentence in enumerate(sentences, start=1):
        print(f"\nProcessing sentence {idx}: {sentence}")
        prompt = generate_image_prompt(paragraph, sentence)
        print(f"Generated prompt: {prompt}")
        imgs = generate_images(prompt, sentence_number=idx, num_images=2)
        all_image_paths.extend(imgs)
    
    # For each generated image, create an effect clip and save it.
    clip_files = []
    for i, img_path in enumerate(all_image_paths):
        effect_func = effect_functions[i % len(effect_functions)]
        clip_file = save_effect_clip(img_path, effect_func, i+1)
        if clip_file:
            clip_files.append(clip_file)
    
    # For each consecutive pair of clips, generate a transition.
    transition_files = []
    for i in range(len(clip_files) - 1):
        last_frame = get_last_frame(clip_files[i])
        first_frame = get_first_frame(clip_files[i+1])
        if last_frame is None or first_frame is None:
            continue
        # Cycle through transitions using the index
        trans_func = transition_functions[i % len(transition_functions)]
        trans_file = save_transition_clip(last_frame, first_frame, trans_func, i+1)
        transition_files.append(trans_file)
    
    # Build final video sequence:
    # Sequence: clip_0, transition_0, clip_1, transition_1, clip_2, ... etc.
    final_sequence = []
    final_sequence.append(clip_files[0])
    for i in range(1, len(clip_files)):
        final_sequence.append(transition_files[i-1])
        final_sequence.append(clip_files[i])
    
    # Merge final video files into one final video.
    merge_video_files(final_sequence, FINAL_VIDEO_PATH)

if __name__ == "__main__":
    main()
