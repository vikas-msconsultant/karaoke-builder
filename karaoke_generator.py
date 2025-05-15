import librosa
import numpy as np
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from pydub import AudioSegment
import os

class KaraokeGenerator:
    def __init__(self):
        self.sample_rate = 44100
        
    def separate_vocals(self, audio_path):
        """Separate vocals from the instrumental using librosa"""
        # Load the audio file
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Compute the spectrogram
        S_full, phase = librosa.magphase(librosa.stft(y))
        
        # Compute soft mask for vocals
        S_filter = librosa.decompose.nn_filter(S_full,
                                             aggregate=np.median,
                                             metric='cosine',
                                             width=int(librosa.time_to_frames(2, sr=sr)))
        
        S_filter = np.minimum(S_full, S_filter)
        margin_i, margin_v = 2, 10
        power = 2
        
        mask_i = librosa.util.softmask(S_filter,
                                     margin_i * (S_full - S_filter),
                                     power=power)
        
        mask_v = librosa.util.softmask(S_full - S_filter,
                                     margin_v * S_filter,
                                     power=power)
        
        # Get instrumental and vocals
        S_foreground = mask_v * S_full
        S_background = mask_i * S_full
        
        # Convert back to audio
        foreground = librosa.istft(S_foreground * phase)
        background = librosa.istft(S_background * phase)
        
        return background, foreground
    
    def create_karaoke_video(self, audio_path, lyrics, output_path):
        """Create a karaoke video with synchronized lyrics"""
        # Separate vocals and instrumental
        instrumental, _ = self.separate_vocals(audio_path)
        
        # Save instrumental temporarily
        librosa.output.write_wav('temp_instrumental.wav', instrumental, self.sample_rate)
        
        # Create video clips for lyrics
        clips = []
        duration = librosa.get_duration(y=instrumental, sr=self.sample_rate)
        
        # Simple text display (you can enhance this with proper timing)
        for i, line in enumerate(lyrics):
            start_time = i * 5  # Simple timing: new line every 5 seconds
            if start_time < duration:
                text_clip = (TextClip(line, fontsize=70, color='white', font='Arial')
                           .set_position('center')
                           .set_start(start_time)
                           .set_duration(5))
                clips.append(text_clip)
        
        # Create black background
        background = ColorClip((1920, 1080), color=(0, 0, 0))
        background = background.set_duration(duration)
        
        # Combine all clips
        final_video = CompositeVideoClip([background] + clips)
        
        # Add instrumental audio
        final_video = final_video.set_audio(AudioFileClip('temp_instrumental.wav'))
        
        # Write final video
        final_video.write_videofile(output_path, fps=24)
        
        # Clean up temporary files
        os.remove('temp_instrumental.wav')

# Example usage
if __name__ == "__main__":
    # Example lyrics (you would normally load these from a file)
    sample_lyrics = [
        "First line of the song",
        "Second line comes here",
        "Third line of lyrics",
        "Final line to sing"
    ]
    
    generator = KaraokeGenerator()
    generator.create_karaoke_video(
        audio_path="input_song.mp3",
        lyrics=sample_lyrics,
        output_path="karaoke_output.mp4"
    )