import asyncio
import os
import re
from pydub import AudioSegment
import edge_tts

# Set your download directory path (for example, on an Android device)
DOWNLOAD_DIR = "/storage/emulated/0/Download/"

async def generate_tts(text: str, output_file: str, voice: str):
    """
    Generate TTS audio for the given text using the specified voice and
    save the output to the given file path.
    """
    communicate = edge_tts.Communicate(text, voice=voice)
    with open(output_file, "wb") as f:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])

async def main():
    # Ask the user to select a voice by number
    print("Select a voice:")
    print("  1: ml-IN-MidhunNeural (Male, Friendly, Positive)")
    print("  2: ml-IN-SobhanaNeural (Female, Friendly, Positive)")
    choice = input("Enter the number corresponding to your chosen voice: ").strip()

    if choice == "1":
        selected_voice = "ml-IN-MidhunNeural"
    elif choice == "2":
        selected_voice = "ml-IN-SobhanaNeural"
    else:
        print("Invalid choice. Defaulting to ml-IN-MidhunNeural.")
        selected_voice = "ml-IN-MidhunNeural"

    # Input text from user
    text = input("Enter the text: ")

    # Split the paragraph by full stop and remove any empty sentences
    sentences = re.split(r'\.\s*', text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    # Create the download directory if it doesn't exist
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)

    audio_files = []

    # Process each sentence individually
    for idx, sentence in enumerate(sentences):
        output_file = os.path.join(DOWNLOAD_DIR, f"sentence_{idx}.mp3")
        print(f"\nProcessing sentence {idx + 1} with voice {selected_voice}:")
        print(f"  \"{sentence}\"")
        
        # Generate TTS audio for the sentence
        await generate_tts(sentence, output_file, selected_voice)
        
        # Load the generated audio to calculate its duration
        audio = AudioSegment.from_file(output_file)
        duration = audio.duration_seconds
        print(f"Sentence {idx + 1} duration: {duration:.2f} seconds")
        
        # Save the filename for later combination
        audio_files.append(output_file)

    # Combine all individual audio clips into one file
    combined_audio = AudioSegment.empty()
    for file in audio_files:
        audio = AudioSegment.from_file(file)
        combined_audio += audio

    combined_file = os.path.join(DOWNLOAD_DIR, "combined_audio.mp3")
    combined_audio.export(combined_file, format="mp3")
    
    # Calculate and print the final combined duration
    final_duration = combined_audio.duration_seconds
    print(f"\nCombined audio saved to: {combined_file}")
    print(f"Final combined duration: {final_duration:.2f} seconds")

    # Delete individual sentence audio files
    for file in audio_files:
        os.remove(file)
    print("Individual audio clips have been deleted.")

if __name__ == "__main__":
    asyncio.run(main())
