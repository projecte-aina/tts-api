class SpeakerException(Exception):
    def __init__(self, speaker_id: str):
        self.speaker_id = speaker_id


class LanguageException(Exception):
    def __init__(self, language: str):
        self.language = language