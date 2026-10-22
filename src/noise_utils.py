import numpy as np

def _as_mono_float(waveform):
    """检查并转换为一维float64波形"""
    x = np.asarray(waveform, dtype=np.float64)  #把 waveform 转成一个 NumPy 数组，并强制其元素类型为 float64

    if x.ndim != 1 or x.size == 0:
        raise ValueError("waveform必须是非空的一维数组")

    if not np.isfinite(x).all():
        raise ValueError("waveform包含NaN或Inf")

    return x

def measure_power(waveform):
    """计算离散波形的均方功率"""
    x = _as_mono_float(waveform)
    return float(np.mean(x ** 2))

def measure_snr_db(clean, noisy):
    """根据clean与noisy的差值测量SNR"""
    x = _as_mono_float(clean)
    y = _as_mono_float(noisy)

    if x.shape != y.shape:
        raise ValueError("clean与noisy的形状必须相同")

    signal_power = measure_power(x)
    noise = y - x
    noise_power = measure_power(noise)

    if signal_power <= 0:
        return ValueError("clean信号功率必须大于0")

    if noise_power == 0:
        return float("inf")

    return float(
        10 * np.log10(signal_power / noise_power)
    )

def add_gaussian_noise_at_snr(clean, target_snr_db=10.0, seed=42):
    """
    按功率定义的目标SNR添加高斯白噪声

    返回：
        noisy:加噪后的波形
        noise:实际添加的噪声
    """
    x = _as_mono_float(clean)

    if not np.isfinite(target_snr_db):
        raise ValueError("目标SNR必须为有限数值")

    signal_power = measure_power(x)

    if signal_power <= 1e-12:
        raise ValueError("信号功率过小，无法可靠设置SNR")

    #1.计算目标噪声功率
    target_noise_power = (
        signal_power / (10 ** (target_snr_db / 10))
    )

    #2.生成指定seed的随机噪声
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(x.shape)

    #去除有限样本产生的直流偏置
    z = z - np.mean(z)

    #3.测量随机噪声的实际功率
    random_noise_power = measure_power(z)

    if random_noise_power <= 0:
        raise ValueError("随机噪声功率异常")

    #4.计算幅度缩放系数
    alpha = np.sqrt(
        target_noise_power / random_noise_power
    )    

    #5.加噪
    noise = alpha * z
    noisy = x + noise

    return noisy, noise

def protect_pair_peak(clean, noisy, ceiling=0.99):
    """
    若峰值过大，则对clean/noisy共同缩放。
    返回缩放后的clean、noisy，以及共同缩放系数。
    """
    x = _as_mono_float(clean)
    y = _as_mono_float(noisy)

    if x.shape != y.shape:
        raise ValueError("clean与noisy的形状必须相同")

    if not 0 < ceiling < 1:
        raise ValueError("ceiling必须在0和1之间")

    peak = max(
        np.max(np.abs(x)),
        np.max(np.abs(y))
    )

    if peak == 0:
        raise ValueError("不能对全零信号执行此操作")

    scale = min(1.0, ceiling / peak)

    return x * scale, y * scale, scale