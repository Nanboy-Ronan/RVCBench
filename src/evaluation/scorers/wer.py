import string
from .text import _contains_cjk, _normalize_zh_text, _to_simplified_zh


class WERScorer:
    version = 'whisper_medium_ascii_punctuation_removed_v2'
    dependencies = ('openai-whisper', 'torch', 'torchaudio', 'jiwer', 'jieba', 'numpy',
                    'opencc', 'opencc-python-reimplemented', 'hanziconv')

    def __init__(self, device, logger):
        self.device, self.logger, self.model = device, logger, None
        self.normalization = 'ascii_punctuation_removed_v2'

    def set_normalization(self, normalization):
        if normalization not in ('lowercase_v1', 'ascii_punctuation_removed_v2'):
            raise ValueError(f'Unsupported WER normalization: {normalization}')
        self.normalization = normalization
        self.version = 'whisper_medium_' + normalization

    def prepare(self):
        import whisper
        self.model = whisper.load_model('medium', device=self.device)
        self.model_provenance = {'model': 'whisper-medium', 'weights_url': whisper._MODELS['medium']}

    def score(self, request):
        import jiwer
        kwargs = {'task': 'transcribe', 'condition_on_previous_text': False, 'without_timestamps': True}
        hint = str(request.language or '').strip().lower()
        if hint and hint not in ('auto', 'none', 'null'):
            kwargs['language'] = hint
        result = self.model.transcribe(str(request.generated), **kwargs)
        transcription = result['text']
        if _contains_cjk(transcription):
            transcription = _to_simplified_zh(transcription)
        reference, hypothesis = str(request.text or ''), str(transcription)
        if _contains_cjk(reference) or result.get('language') == 'zh':
            import jieba
            reference = ' '.join(jieba.lcut(_normalize_zh_text(reference)))
            hypothesis = ' '.join(jieba.lcut(_normalize_zh_text(hypothesis)))
        else:
            reference = normalize_english(reference, self.normalization)
            hypothesis = normalize_english(hypothesis, self.normalization)
        return {'wer': float(jiwer.wer(reference, hypothesis)), 'predicted_text': transcription}

    def close(self):
        self.model = None


def normalize_english(text, normalization):
    if normalization == 'ascii_punctuation_removed_v2':
        text = text.translate(str.maketrans('', '', string.punctuation))
    elif normalization != 'lowercase_v1':
        raise ValueError(f'Unsupported WER normalization: {normalization}')
    return text.lower()
