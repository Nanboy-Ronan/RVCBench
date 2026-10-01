class MCDScorer:
    version = 'pymcd_dtw_v1'
    dependencies = ('pymcd', 'librosa', 'numpy', 'scipy', 'numba')

    def __init__(self, device, logger):
        self.model = None

    def prepare(self):
        from pymcd.mcd import Calculate_MCD
        self.model = Calculate_MCD(MCD_mode='dtw')
        self.model_provenance = {'implementation': 'pymcd', 'MCD_mode': 'dtw'}

    def score(self, request):
        return {'mcd': float(self.model.calculate_mcd(str(request.reference), str(request.generated)))}

    def close(self):
        self.model = None
