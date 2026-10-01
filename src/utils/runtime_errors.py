"""Errors after which CUDA observations from the same process are unreliable."""


def invalid_cuda_context(error):
    message = str(error).lower()
    return any(marker in message for marker in (
        'device-side assert', 'illegal memory access',
        'cuda_error_assert', 'cuda_error_illegal_address'))
