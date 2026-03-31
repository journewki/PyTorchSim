import torch
import torch._dynamo

def test_result(name, out, cpu_out, rtol=1e-3, atol=1e-3):
    if torch.allclose(out.cpu(), cpu_out, rtol=rtol, atol=atol):
        message = f"|{name} Test Passed|"
        print("-" * len(message))
        print(message)
        print("-" * len(message))
    else:
        message = f"|{name} Test Failed|"
        print("-" * len(message))
        print(message)
        print("-" * len(message))
        print("custom out: ", out.cpu())
        print("cpu out: ", cpu_out)
        max_diff = torch.max(torch.abs(out.cpu() - cpu_out))
        print("Max diff: ", max_diff)
        exit(1)

def test_depthwise_conv2d(device, batch_size=1, channels=32, input_size=16, kernel_size=3, stride=1, padding=1, name=None):
    def custom_depthwise_conv2d(x, w, bias):
        groups = x.shape[1]
        conv = torch.nn.Conv2d(groups, groups, kernel_size=w.shape[-1], stride=stride, padding=padding, groups=groups, bias=True)
        conv.weight = torch.nn.Parameter(w)
        conv.bias = torch.nn.Parameter(bias)
        return conv(x)

    torch.manual_seed(0)
    x = torch.randn(batch_size, channels, input_size, input_size).to(device=device)
    w = torch.randn(channels, 1, kernel_size, kernel_size).to(device=device)
    bias = torch.randn(channels).to(device=device)

    opt_fn = torch.compile(dynamic=False)(custom_depthwise_conv2d)
    res = opt_fn(x, w, bias)
    ref = custom_depthwise_conv2d(x.cpu(), w.cpu(), bias.cpu())

    test_name = name or f"DepthwiseConv2d C={channels} S={input_size} K={kernel_size} stride={stride}"
    test_result(test_name, res, ref, rtol=1e-3, atol=1e-3)
    print("Max diff:", torch.max(torch.abs(res.cpu() - ref)).item())

if __name__ == "__main__":
    device = torch.device("npu:0")
    torch._dynamo.config.cache_size_limit = 64
    with torch.no_grad():
        test_depthwise_conv2d(device, channels=4, input_size=8, kernel_size=3, padding=1, name="Tiny depthwise 4ch")
        test_depthwise_conv2d(device, channels=32, input_size=16, kernel_size=3, padding=1, name="Depthwise 32ch 16x16 K3")
        test_depthwise_conv2d(device, channels=128, input_size=8, kernel_size=3, padding=1, name="Depthwise 128ch 8x8 K3")
        test_depthwise_conv2d(device, channels=320, input_size=32, kernel_size=3, padding=1, name="GLUMBConv-like 320ch 32x32 K3")
