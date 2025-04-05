import pollinations
from g4f.client import Client
import os
import random

# Initialize APIs
client = Client()
download_dir = "/storage/emulated/0/Download/AI_Images"
os.makedirs(download_dir, exist_ok=True)

# Get user input and split sentences [[1]]
text = input("Enter a paragraph: ")
sentences = [s.strip() for s in text.split('.') if s.strip()]

for index, sentence in enumerate(sentences, 1):
    # Generate image prompt using G4F [[2]][[4]]
    prompt_request = f"Create a detailed DALL-E 3 style prompt in english for: '{sentence}'"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt_request}],
        web_search=False
    )
    image_prompt = response.choices[0].message.content.strip()
    
    print(f"\nSentence {index}: {sentence}")
    print(f"Image Prompt: {image_prompt}")
    
    # Generate 2 enhanced images per prompt [[6]][[8]]
    for img_num in [1, 2]:
        try:
            # Configure Pollinations with enhancement [[9]]
            image_model = pollinations.Image(
                model="flux_3d",
                seed="random",  # Random seed for variation
                width=1024,
                height=1024,
                enhance=True,  # Quality enhancement enabled
                nologo=True,
                private=True,
                safe=False,
                referrer="pollinations.py"
            )
            
            # Generate and save image
            image = image_model(prompt=image_prompt)
            filename = f"img_{index}_{img_num}.png"
            file_path = os.path.join(download_dir, filename)
            
            with open(file_path, "wb") as f:
                f.write(image.data)
            
            print(f"✓ Image {img_num} saved: {file_path}")
        except Exception as e:
            print(f"✗ Error: {str(e)}")
