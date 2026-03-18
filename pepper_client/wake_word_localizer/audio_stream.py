class WakeWordAudioService(object):
    """
    qi service that receives raw audio from Pepper's ALAudioDevice
    and pushes each chunk into SharedState for the transcriber.

    Follows the pattern from pepper_api/audio.py (AudioManager2):
    - Registered via session.registerService()
    - processRemote callback invoked by ALAudioDevice
    - subscribe/unsubscribe lifecycle
    """

    def __init__(self, shared_state):
        super(WakeWordAudioService, self).__init__()
        self.module_name = "WakeWordAudioService"
        self.shared_state = shared_state
        self.sample_rate = 16000
        self.channels = 1

    def init_service(self, session):
        self.audio_service = session.service("ALAudioDevice")

    def start(self):
        self.audio_service.setClientPreferences(
            self.module_name, self.sample_rate, self.channels, 0
        )
        self.audio_service.subscribe(self.module_name)

    def stop(self):
        try:
            self.audio_service.unsubscribe(self.module_name)
        except Exception:
            pass

    def processRemote(self, nbOfChannels, nbOfSamplesByChannel, timeStamp, inputBuffer):
        """Called by ALAudioDevice on each audio chunk.
        Must return quickly - only does a deque append (O(1))."""
        if self.shared_state.running:
            self.shared_state.push_audio(bytes(inputBuffer))
