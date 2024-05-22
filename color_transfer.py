
import batorch as bt
import micomputing as mc

def Sigma(source: bt.Tensor, target: bt.Tensor, sig=0, mean=None):
    if mean is None: mean = lambda x: mc.blur(x)
    Dsource = bt.grad_image(source, pad=True)
    Dtarget = bt.grad_image(target, pad=True)
    Msource = mean(source)
    Mtarget = mean(target)
    point_Sigma11 = mean((Dsource * Dsource).sum([]))
    point_Sigma12 = mean((Dsource * Dtarget).sum([]))
    point_Sigma22 = mean((Dtarget * Dtarget).sum([]))
    local_Sigma11 = mean(source * source) - Msource ** 2
    local_Sigma12 = mean(source * target) - Msource * Mtarget
    local_Sigma22 = mean(target * target) - Mtarget ** 2
    Sigma11 = sig * point_Sigma11 + local_Sigma11
    Sigma12 = sig * point_Sigma12 + local_Sigma12
    Sigma22 = sig * point_Sigma22 + local_Sigma22
    return Sigma11, Sigma12, Sigma22

def color_transfer(source: bt.Tensor, target: bt.Tensor, sig=0):
    mean = lambda x: mc.blur(x) #, kernel_size=7)
    Sigma11, Sigma12, Sigma22 = Sigma(source=source, target=target, sig=sig, mean=mean)
    Delta = (Sigma11 - Sigma22) ** 2 + 4 * Sigma12 ** 2
    Kappa = (Sigma11 - Sigma22) / (Delta + bt.eps).sqrt()
    return bt.sign(Sigma12) * (1 - Kappa).sqrt() * (source - mean(source)) / (1+Kappa).sqrt().clamp(bt.eps) + mean(target)

def LocalIntensityCorrelation(source: bt.Tensor, target: bt.Tensor, sig=0):
    Sigma11, Sigma12, Sigma22 = Sigma(source=source, target=target, sig=sig)
    return Sigma12.abs() / (Sigma11 * Sigma22).clamp(bt.eps).sqrt()
    