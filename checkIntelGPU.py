import intel_extension_for_pytorch as ipex
import torch

print("IPEX のバージョン:", ipex.__version__)
print("XPU (Intel GPU) 利用可能:", torch.xpu.is_available())
print("利用可能なデバイス:", torch.xpu.device_count())
print("使用中のデバイス:", torch.xpu.current_device())
print("デバイス名:", torch.xpu.get_device_name(0))
print("メモリ使用量:", torch.xpu.memory_allocated(0) / 1024 / 1024, "MB")
