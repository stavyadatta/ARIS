import numpy as np
import logging

logger = logging.getLogger(__name__)

# ALAudioDevice channel constants (from NAOqi SDK)
AL_ALLCHANNELS = 0
AL_LEFTCHANNEL = 1
AL_RIGHTCHANNEL = 2
AL_FRONTCHANNEL = 3
AL_REARCHANNEL = 4

# 4-channel audio is only available at 48 kHz
SRP_SAMPLE_RATE = 48000
SRP_N_CHANNELS = 4


class SRPAudioService(object):
    """qi service that captures 4-channel (all microphones) audio from
    Pepper's ALAudioDevice at 48 kHz for SRP-PHAT processing.

    This runs alongside the existing WakeWordAudioService (which grabs
    1-channel / 16 kHz for Vosk).  ALAudioDevice supports multiple
    simultaneous subscribers with different format preferences.

    Audio flow
    ----------
    ALAudioDevice  --processRemote-->  this service
        --> converts raw bytes to (n_samples, 4) float32 numpy array
        --> pushes into SharedState.srp_audio queue
        --> SoundTracker thread pops & runs SRP-PHAT
    """

    def __init__(self, shared_state):
        super(SRPAudioService, self).__init__()
        self.module_name = "SRPAudioService"
        self.shared_state = shared_state
        self.sample_rate = SRP_SAMPLE_RATE

    def init_service(self, session):
        self.audio_service = session.service("ALAudioDevice")

    def start(self):
        # Request all 4 channels, 48 kHz, deinterleaved
        # Deinterleaved (1) means the buffer layout is:
        #   [ch0_s0, ch0_s1, ..., ch0_sN,  ch1_s0, ..., ch3_sN]
        # which is easier to reshape into (n_samples, 4)
        self.audio_service.setClientPreferences(
            self.module_name,
            self.sample_rate,
            AL_ALLCHANNELS,   # 0 = all four microphones
            1,                # 1 = deinterleaved
        )
        self.audio_service.subscribe(self.module_name)
        logger.info(
            "SRP audio subscribed: %d Hz, %d channels, deinterleaved",
            self.sample_rate, SRP_N_CHANNELS,
        )

    def stop(self):
        try:
            self.audio_service.unsubscribe(self.module_name)
        except Exception:
            pass

    def processRemote(self, nbOfChannels, nbOfSamplesByChannel,
                      timeStamp, inputBuffer):
        """Called by ALAudioDevice on every audio buffer (~170 ms).

        Must return quickly — we just convert and enqueue.

        With deinterleaved layout the raw buffer (int16) is arranged as:
            [FL_0 FL_1 ... FL_N  FR_0 ... FR_N  RL_0 ... RL_N  RR_0 ... RR_N]
        We reshape to (N, 4) so column k is microphone k.
        """
        if not self.shared_state.running:
            return

        try:
            audio = np.frombuffer(inputBuffer, dtype=np.int16)

            n_ch = int(nbOfChannels)
            n_samp = int(nbOfSamplesByChannel)

            if n_ch < SRP_N_CHANNELS:
                return   # unexpected format

            # Deinterleaved: reshape to (n_channels, n_samples) then transpose
            multichannel = audio.reshape((n_ch, n_samp)).T   # (n_samp, n_ch)

            # Normalise int16 -> float32 in [-1, 1]
            multichannel = multichannel.astype(np.float32) / 32768.0

            self.shared_state.push_srp_audio(multichannel)

        except Exception as e:
            logger.warning("SRP audio processing error: %s", e)
