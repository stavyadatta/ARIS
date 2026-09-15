import os
import torch
import wave
import whisper
import tempfile
import atexit

from turn_timing import span

class _WhisperSpeech2Text:
    def __init__(self, model_name="large-v3"):
        """
        Initialize the Whisper model for speech-to-text.
        """
        self.model = whisper.load_model(model_name, device=torch.device("cuda:0"))

    def _save2temp(self, audio_data):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file_path = temp_file.name
        temp_file.close()
        encoding_map = {
            "PCM_8": 1,   # 8-bit audio
            "PCM_16": 2,  # 16-bit audio
            "PCM_24": 3,  # 24-bit audio
            "PCM_32": 4   # 32-bit audio
        }
        sample_width = encoding_map.get(audio_data["encoding"])
        try:
            with wave.open(temp_file.name, 'wb') as wave_file:
                wave_file.setnchannels(audio_data["num_channels"])
                wave_file.setsampwidth(sample_width) 
                wave_file.setframerate(audio_data["sample_rate"])
                wave_file.writeframes(audio_data["audio_data"])


            atexit.register(lambda: os.remove(temp_file_path) if os.path.exists(temp_file_path) else None)
            return temp_file.name
        except Exception as e:
            print(f"Error saving audio to file: {e}")
            raise

    def __call__(self, audio_img_data):
        """
        Transcribe speech from audio data to text.
        :param audio_data: Data Object
        :return: Transcribed text.
        """
        # Getting temp file to write the audio bytes into the
        # wav format
        with span("whisper.write_wav"):
            temp_recording = self._save2temp(audio_img_data)

        # Convert bytes to a torch audio tensor
        with span("whisper.load_audio"):
            audio_tensor = torch.tensor(
                whisper.pad_or_trim(whisper.load_audio(temp_recording))
            ).float()
        # Run Whisper to transcribe the audio
        with span("whisper.transcribe"):
            result = self.model.transcribe(audio_tensor, language='en')
        return result["text"]

