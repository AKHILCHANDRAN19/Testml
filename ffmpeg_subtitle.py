import os
import subprocess
import shutil
import math
import json # To parse ffprobe's JSON output
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# -----------------------------
# Configuration and File Paths
# -----------------------------
input_video_path = "/storage/emulated/0/Download/final_output.mp4"
# Ensure this font file exists at this exact path
font_path = "/storage/emulated/0/Download/AnekMalayalam-Bold.ttf"
# Using a distinct name for the output of this script version
output_video_path = "/storage/emulated/0/Download/output_with_text_final.mp4"
# Folder for temporary images (will be created and deleted)
temp_frame_folder = "/storage/emulated/0/Download/temp_frames"

# Cycle of high-intensity colors: white → yellow → green → repeat
color_cycle = ["#FFFFFF", "#FFFF00", "#00FF00"]

# --- Main Execution Block ---
try: # Wrap main logic in try/finally for cleanup

    # -----------------------------
    # Get User Input
    # -----------------------------
    user_text = input("Enter the text to add to the video: ")
    if not user_text:
        print("No text entered. Exiting.")
        exit()

    try:
        user_input_size = input("Enter font size (default is 50): ").strip()
        font_size = int(user_input_size) if user_input_size else 50
    except ValueError:
        font_size = 50
        print("Invalid input. Using default font size of 50.")

    # -----------------------------
    # Get Video Information using command-line ffprobe
    # -----------------------------
    print(f"Getting video info for: {input_video_path}")
    if not os.path.exists(input_video_path):
        print(f"Error: Input video not found at {input_video_path}")
        exit()

    try:
        # Construct the ffprobe command
        ffprobe_cmd = [
            'ffprobe',
            '-v', 'quiet',                  # Suppress logging
            '-print_format', 'json',        # Output format
            '-show_format',                 # Include format info (duration)
            '-show_streams',                # Include stream info (width, height, fps)
            input_video_path
        ]

        print(f"Running ffprobe command: {' '.join(ffprobe_cmd)}")
        result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, check=True)
        metadata = json.loads(result.stdout) # Parse the JSON output

        # Find the video stream and extract info
        video_stream = None
        for stream in metadata.get('streams', []):
            if stream.get('codec_type') == 'video':
                video_stream = stream
                break

        if not video_stream:
            raise ValueError("Error: No video stream found in the ffprobe output.")

        video_width = int(video_stream.get('width', 0))
        video_height = int(video_stream.get('height', 0))

        if video_width == 0 or video_height == 0:
             raise ValueError("Error: Could not determine video dimensions (width/height).")

        # Get duration - check format first, then stream
        duration_str = metadata.get('format', {}).get('duration')
        if duration_str is None:
            duration_str = video_stream.get('duration') # Fallback to stream duration

        if duration_str is None:
             raise ValueError("Error: Could not determine video duration.")
        video_duration = float(duration_str)

        # Get FPS - handle fraction format (e.g., "30/1", "2997/100")
        fps_str = video_stream.get('avg_frame_rate')
        if not fps_str or '/' not in fps_str:
            fps_str = video_stream.get('r_frame_rate') # Try r_frame_rate as fallback
            if not fps_str or '/' not in fps_str:
                 print("Warning: Could not reliably determine FPS from ffprobe. Assuming 24.")
                 video_fps = 24.0
            else:
                num, den = map(int, fps_str.split('/'))
                video_fps = num / den if den != 0 else 24.0
        else:
            num, den = map(int, fps_str.split('/'))
            video_fps = num / den if den != 0 else 24.0 # Default to 24 if denominator is 0

        total_frames = int(math.ceil(video_duration * video_fps)) # Use ceil

        print(f"Video Info: {video_width}x{video_height}, Duration: {video_duration:.2f}s, FPS: {video_fps:.2f}, Total Frames: {total_frames}")

    except FileNotFoundError:
        print("\nError: 'ffprobe' command not found. Make sure FFmpeg (which includes ffprobe) is installed and in your system's PATH.")
        exit()
    except subprocess.CalledProcessError as e:
        print(f"\nError running ffprobe. Return code: {e.returncode}")
        print(f"Stderr: {e.stderr}")
        exit()
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"\nError processing ffprobe output: {e}")
        exit()
    except Exception as e:
        print(f"\nAn unexpected error occurred while getting video info: {e}")
        exit()


    # -----------------------------
    # Prepare Text and Folders
    # -----------------------------
    words = user_text.split()
    if not words:
        print("Text contains no words. Exiting.")
        exit()

    # Create temporary folder for frames - ensure it's clean
    if os.path.exists(temp_frame_folder):
        print(f"Removing existing temporary folder: {temp_frame_folder}")
        shutil.rmtree(temp_frame_folder)
    os.makedirs(temp_frame_folder)
    print(f"Created temporary folder: {temp_frame_folder}")

    # -----------------------------
    # Load Font
    # -----------------------------
    try:
        # Try loading font with RAQM layout for complex scripts
        font = ImageFont.truetype(font_path, font_size, layout_engine=ImageFont.Layout.RAQM)
        print(f"Font loaded successfully with RAQM layout: {font_path}")
    except ImportError: # Handle if RAQM support isn't fully available
         print("Warning: RAQM layout engine not found (install 'raqm' library for best complex script support). Trying default layout.")
         try:
             font = ImageFont.truetype(font_path, font_size)
             print(f"Font loaded successfully with default layout: {font_path}")
         except IOError:
            print(f"Error: Font file not found or cannot be opened at {font_path}.")
            print("Using default PIL font (may not render specific characters correctly).")
            font = ImageFont.load_default()
    except IOError:
        print(f"Error: Font file not found or cannot be opened at {font_path}.")
        print("Using default PIL font (may not render specific characters correctly).")
        font = ImageFont.load_default()
    except Exception as e:
         print(f"Error loading font: {e}")
         print("Using default PIL font.")
         font = ImageFont.load_default()

    # -----------------------------
    # Function to Create a Single Text Frame
    # -----------------------------
    def create_text_frame(frame_index, total_frames, video_duration, video_fps, words, font, font_size, video_width, video_height):
        """
        Generates a transparent PNG image with the correctly styled word for a given frame.
        """
        total_words = len(words)
        # Calculate time per word, prevent division by zero if video_duration is 0
        time_per_word = (video_duration / total_words) if (total_words > 0 and video_duration > 0) else video_duration
        # Calculate current time based on frame index and fps
        current_time = frame_index / video_fps if video_fps > 0 else 0

        # Determine which word to display
        current_word_index = -1
        if time_per_word > 0:
            current_word_index = min(int(current_time // time_per_word), total_words - 1)
        elif total_words > 0 : # If duration is 0 or negative, show the first word always
             current_word_index = 0


        # Create a blank transparent image matching video dimensions
        frame_image = Image.new("RGBA", (video_width, video_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(frame_image)

        if current_word_index != -1:
            current_word = words[current_word_index]
            text_color = color_cycle[current_word_index % len(color_cycle)]

            # --- Text Styling ---
            stroke_width = max(2, font_size // 15) # Scale stroke with font size (min 2)
            shadow_offset = (max(1, font_size // 20), max(1, font_size // 20)) # Scale shadow with font size (min 1)
            shadow_color = "black"
            stroke_color = "black"

            # Calculate text bounding box *including stroke* to get accurate size
            try:
                # Use textbbox for more accurate sizing including descenders/ascenders
                bbox = draw.textbbox((0, 0), current_word, font=font, stroke_width=stroke_width)
                # bbox is (left, top, right, bottom)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                # Base position for drawing (top-left corner before centering)
                draw_x_base = bbox[0]
                draw_y_base = bbox[1]

            except Exception as e:
                print(f"Warning: Could not get bbox for '{current_word}' (maybe empty?): {e}")
                text_width, text_height = 0, 0
                draw_x_base, draw_y_base = 0, 0

            if text_width > 0 and text_height > 0:
                # Calculate position for centering based on the bounding box dimensions
                text_x = (video_width - text_width) // 2
                text_y = (video_height - text_height) // 2

                # Adjust the final draw position by the bbox's top-left offset
                # This ensures the visual center aligns with the calculated center
                final_draw_x = text_x - draw_x_base
                final_draw_y = text_y - draw_y_base

                # Calculate shadow position relative to the final draw position
                shadow_x = final_draw_x + shadow_offset[0]
                shadow_y = final_draw_y + shadow_offset[1]

                # 1. Draw Shadow (text with stroke, both in shadow color)
                draw.text(
                    (shadow_x, shadow_y),
                    current_word,
                    font=font,
                    fill=shadow_color, # Shadow fill
                    stroke_width=stroke_width,
                    stroke_fill=shadow_color # Shadow stroke
                )

                # 2. Draw Main Text with Stroke
                draw.text(
                    (final_draw_x, final_draw_y),
                    current_word,
                    font=font,
                    fill=text_color,         # Main text color
                    stroke_width=stroke_width,
                    stroke_fill=stroke_color # Outline color
                )
            # else: (Optional: print if skipping word draw)
            #      print(f"Skipping drawing for word '{current_word}' due to zero size.")

        # Save the frame as a PNG image (preserves transparency)
        frame_filename = os.path.join(temp_frame_folder, f"frame_{frame_index:06d}.png")
        try:
            frame_image.save(frame_filename)
        except Exception as e:
            print(f"Error saving frame {frame_index}: {e}")
            # Decide if you want to stop or just continue
            # raise e # Uncomment to stop on save error

        # return frame_filename # Return filename if needed elsewhere

    # -----------------------------
    # Generate All Text Frames
    # -----------------------------
    print(f"Generating {total_frames} text frames...")
    for i in range(total_frames):
        # Pass video_fps to the frame generation function
        create_text_frame(i, total_frames, video_duration, video_fps, words, font, font_size, video_width, video_height)
        if (i + 1) % 50 == 0 or (i + 1) == total_frames: # Print progress
            print(f"  Generated frame {i+1}/{total_frames}")

    print("Finished generating text frames.")

    # -----------------------------
    # Use FFmpeg to Overlay Frames onto Video
    # -----------------------------
    print("Starting FFmpeg process to overlay text...")

    ffmpeg_cmd = [
        'ffmpeg',
        '-y',  # Overwrite output file without asking
        '-i', input_video_path,  # Input 0: Original video
        '-framerate', str(video_fps), # Specify FPS for the image sequence input
        '-i', os.path.join(temp_frame_folder, 'frame_%06d.png'), # Input 1: Image sequence
        # Filter complex: Set timebases and overlay, pass EOF to avoid premature ending
        '-filter_complex', f'[0:v]settb=AVTB[base];[1:v]settb=AVTB[overlay];[base][overlay]overlay=x=0:y=0:eof_action=pass[outv]',
        '-map', '[outv]',  # Map the filtered video stream to output
        '-map', '0:a?',    # Map audio from original video (if it exists)
        '-c:v', 'libx264', # Video codec
        '-preset', 'medium', # Encoding speed/compression tradeoff (use 'ultrafast' for faster testing)
        '-crf', '23',      # Constant Rate Factor (quality, lower value = higher quality)
        '-pix_fmt', 'yuv420p', # Pixel format for compatibility
        '-c:a', 'aac', # Audio codec (standard compatible choice)
        '-b:a', '128k', # Audio bitrate if re-encoding with aac
        # Use '-c:a', 'copy', # (Instead of aac and b:a) to directly copy audio (faster, lossless but potentially less compatible if complex filters were used)
        '-movflags', '+faststart', # Optimize MP4 for web streaming
        output_video_path # Output file path
    ]

    print("\nExecuting FFmpeg command:")
    print(" ".join(ffmpeg_cmd)) # Print the command for debugging

    try:
        # Corrected subprocess call: Only use capture_output=True
        process = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)

        # Print ffmpeg output for feedback upon success
        print("\nFFmpeg process completed successfully.")
        # Uncomment below lines if you want to see potentially long ffmpeg output even on success
        # print("\nFFmpeg stdout:")
        # print(process.stdout)
        # print("\nFFmpeg stderr:") # Often contains progress/details
        # print(process.stderr)
        print(f"\nVideo successfully created: {output_video_path}")

    except subprocess.CalledProcessError as e:
        print(f"\nError during FFmpeg execution. Return code: {e.returncode}")
        print("--- FFmpeg stdout: ---")
        print(e.stdout)
        print("--- FFmpeg stderr: ---") # Error details are often here
        print(e.stderr)
        print("----------------------")
    except FileNotFoundError:
        print("\nError: 'ffmpeg' command not found. Make sure FFmpeg is installed and in your PATH.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during FFmpeg execution: {e}")

# -----------------------------
# Clean Up Temporary Files
# -----------------------------
finally:
    if os.path.exists(temp_frame_folder):
        print(f"\nAttempting to remove temporary frame folder: {temp_frame_folder}")
        try:
            shutil.rmtree(temp_frame_folder)
            print("Temporary files cleaned up successfully.")
        except Exception as e:
            print(f"Warning: Could not remove temporary folder {temp_frame_folder}. You may need to remove it manually. Error: {e}")
