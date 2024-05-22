
try: from pycamia import info_manager
except ImportError: print("Warning: pycamia not loaded. ")

__info__ = info_manager(
    project = "Intensity Distribution",
    package = "Experiment 1",
    author = "Anonymous", 
    create = "2024-03",
    fileinfo = "The recreation of Experiment 1: comparison between LIC and GASD. ",
    requires = ["pycamia", "batorch", "micomputing"]
)

import os, sys, re
import batorch as bt
import micomputing as mc
from datetime import datetime
from pycamia import (
    args, scope, Jump, 
    Path, logging
)

sys.path.append(Path.pardir.abs)
from color_transfer import color_transfer, LocalIntensityCorrelation

args.seed = 1234
args.loss = "NVI"
args.n_dim = 3
args.n_batch = 2
args.n_iter = 80
args.sig_p = 0.
args.noise = 0.
args.down_sample = 2
args.exp = "T1_GASD_sig0"

Experiments = dict(
    # initial: HD [8.018148 ± 1.887841], MSD [0.817116 ± 0.160937], DSC [0.715843 ± 0.099556]
    T4_NVI = "--loss NVI --n_batch 1 --n_iter 200", 
    #       => HD [11.013476 ± 5.482242], MSD [0.321848 ± 0.074095], DSC [0.885016 ± 0.046347]
    T4_MI = "--loss MI --n_batch 1 --n_iter 200", 
    #       => HD [32.807190 ± 5.052977], MSD [0.733406 ± 0.301723], DSC [0.793702 ± 0.118322]
    T4_NMI = "--loss NMI --n_batch 1 --n_iter 200", 
    #       => HD [18.026728 ± 3.543395], MSD [0.387814 ± 0.056801], DSC [0.872230 ± 0.034931]
    T4_NCC = "--loss NCC --n_batch 1 --n_iter 200", 
    #       => HD [7.399347 ± 2.423739], MSD [0.424363 ± 0.070408], DSC [0.848834 ± 0.050483]
    T4_NVI_local = "--loss NVI_local --n_batch 1 --n_iter 200", 
    #       => HD [8.847121 ± 2.567617], MSD [0.322819 ± 0.076096], DSC [0.884573 ± 0.047016]
    T4_MI_local = "--loss MI_local --n_batch 1 --n_iter 200", 
    #       => HD [15.290844 ± 1.133643], MSD [0.636760 ± 0.158216], DSC [0.780423 ± 0.080373]
    T4_NCC_local = "--loss NCC_local --n_batch 1 --n_iter 200", 
    #       => HD [7.284964 ± 2.528624], MSD [0.354609 ± 0.091172], DSC [0.873110 ± 0.054107]
    T4_LIC = "--loss LIC --n_batch 1 --n_iter 200", 
    #       => HD [7.610430 ± 2.263193], MSD [0.325854 ± 0.060042], DSC [0.883378 ± 0.040468]
    T4_GASD = "--loss GASD --n_batch 1 --n_iter 200", 
    #       => HD [18.865503 ± 1.462579], MSD [0.592251 ± 0.144254], DSC [0.806939 ± 0.071984]
    T4_GASD_only = "--loss GASD_only --n_batch 1 --n_iter 200", 
    #       => HD [6.917958 ± 2.865797], MSD [0.644486 ± 0.086996], DSC [0.773811 ± 0.065111]
    T4_SSD = "--loss SSD --n_batch 1 --n_iter 200", 
    #       => HD [6.917958 ± 2.865797], MSD [0.645278 ± 0.087284], DSC [0.773463 ± 0.065318]
    T4_GASD_sig = "--loss GASD_only --n_batch 1 --n_iter 200 --sig_p 0.1", 
    #       => HD [6.917958 ± 2.865797], MSD [0.644486 ± 0.086996], DSC [0.773811 ± 0.065111]
)

exp_name = args.exp
args.parse_attempt(Experiments[exp_name].split())
args.exp = exp_name

bt.manual_seed(args.seed)
dataset_dir = Path.pardir / "brainweb"
output_dir = Path.curdir.mkdir()

def get_data(k=args.down_sample):
    image_t1 = mc.IMG(dataset_dir / "t1.nii.gz")
    image_t2 = mc.IMG(dataset_dir / "t1.nii.gz")
    image_label = mc.IMG(dataset_dir / "atlasLabel.nii.gz")
    template_t1, template_t2, template_label = (
        image_t1.to_tensor().float().duplicate(args.n_batch, {}) / 150,
        image_t2.to_tensor().float().duplicate(args.n_batch, {}) / 150,
        image_label.to_tensor().duplicate(args.n_batch, {})
    )
    template_t1 = template_t1 + args.noise * bt.randn_like(template_t1)
    template_t2 = template_t2 + args.noise * bt.randn_like(template_t2)
    template_t1 = template_t1[..., *(slice(None, None, k),) * args.n_dim].clamp(0, 1)
    template_t2 = template_t2[..., *(slice(None, None, k),) * args.n_dim].clamp(0, 1)
    template_label = template_label[..., *(slice(None, None, k),) * args.n_dim]
    init_trans = mc.rand_DDF(template_t1, std=0.1) @ mc.rand_FFD(template_t1, spacing=1, std=0.5) @ mc.rand_FFD(template_t1, spacing=5, std=2)
    init_trans = mc.VF(init_trans.to_DDF(template_t1))
    warped_t2 = mc.interpolation(template_t2, init_trans).clamp(0, 1)
    warped_label = mc.interpolation(template_label, init_trans, method="Nearest")
    return image_t1.orientation, (template_t1, template_label), (warped_t2, warped_label)

orient, (image_t1, label_t1), (image_t2, label_t2) = get_data()

with scope("plot init images"), Jump(False):
    mc.plt.figure()
    mc.plt.subplots(4)
    mc.plt.imsshow(image_t1, image_t2, label_t1, label_t2, as_orient=orient)
    mc.plt.savefig(output_dir / f"{args.exp}_transformed_images.png")

def GASD_only_loss(i1, i2):
    kernel = bt.gaussian_kernel(n_dim = args.n_dim, kernel_size = 3).unsqueeze([]).duplicate(args.n_batch, {})
    mean = lambda x: eval("bt.nn.functional.conv%dd"%args.n_dim)(x.unsqueeze([1]), kernel, padding = 1).squeeze([])
    return - mean((i2 - i1) ** 2).mean()

def GASD_loss(i1, i2):
    kernel = bt.gaussian_kernel(n_dim = args.n_dim, kernel_size = 3).unsqueeze([]).duplicate(args.n_batch, {})
    mean = lambda x: eval("bt.nn.functional.conv%dd"%args.n_dim)(x.unsqueeze([1]), kernel, padding = 1).squeeze([])
    translated_i1 = color_transfer(i1, i2, args.sig_p)
    return - mean((i2 - translated_i1) ** 2).mean()

def CTr_LIC_loss(i1, i2):
    kernel = bt.gaussian_kernel(n_dim = args.n_dim, kernel_size = 3).unsqueeze([]).duplicate(args.n_batch, {})
    mean = lambda x: eval("bt.nn.functional.conv%dd"%args.n_dim)(x.unsqueeze([1]), kernel, padding = 1).squeeze([])
    translated_i1 = color_transfer(i1, i2, args.sig_p).detach()
    return LocalIntensityCorrelation(translated_i1, i2, args.sig_p)

sim_func = dict(
    LIC = lambda i1, i2: LocalIntensityCorrelation(i1, i2, args.sig_p), 
    NVI = mc.NormalizedVectorInformation, 
    MI = mc.MutualInformation, 
    NMI = mc.NormalizedMutualInformation, 
    NCC = mc.NormalizedCrossCorrelation, 
    NVI_local = lambda i1, i2: mc.Cos2Theta(i1, i2).clamp(min=bt.eps).mean(), 
    MI_local = lambda i1, i2: - bt.log(1 - mc.Cos2Theta(i1, i2).clamp(max=1-bt.eps)).mean() / 2, 
    NCC_local = lambda i1, i2: mc.Cos2Theta(i1, i2).clamp(min=bt.eps).sqrt().mean(), 
    GASD_only = GASD_only_loss, 
    GASD = GASD_loss, 
    SSD = lambda i1, i2: - ((i2 - i1) ** 2).mean(), 
    CTr_LIC = CTr_LIC_loss
)

disp = mc.rand_DDF(image_t1, std=1).offsets.requires_grad_(True)
optimizer = bt.Optimization(bt.optim.Adam, [disp], lr=lambda i: 1 * 0.6 ** (i / args.n_iter))
with logging(log_path=output_dir/f"{Path(__file__).name}_{args.exp}_{datetime.now().strftime('%Y%m%d%H%M%S')}.log"):
    
    for i in range(args.n_iter):
        # bt.summary(disp).show()
        trans = mc.DDF(disp)
        # Y = trans(bt.image_grid(*image_t2.space))
        # warped_image_t2 = image_t2
        # warped_image_t2 = mc.interpolation(image_t2, target_space=bt.image_grid(*image_t2.space) + disp)
        warped_image_t2 = mc.interpolation(image_t2, trans).clamp(0, 1)
        warped_label_t2 = mc.interpolation(label_t2, trans, method='Nearest')
        
        similarity = sim_func[args.loss](image_t1, warped_image_t2)
        bend_coeff = 1e-2 if args.loss in ('GASD',) else 1
        loss = - similarity.mean() + bend_coeff * mc.bending(disp).mean()
        opt = optimizer.new_iter()
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        dsc = mc.LabelDiceScoreCoefficient(label_t1.long(), warped_label_t2.long())
        print(f"Iteration {i + 1}: similarity = {similarity.mean().item():.2e}, loss = {loss.item():.2e}, dice = {dsc.mean().item():.2e}")

    with scope("plot registration results"), Jump(False):
        mc.plt.figure()
        mc.plt.subplots(2)
        mc.plt.maskshow(warped_label_t2==100, warped_label_t2==200, warped_label_t2==300, on=image_t2)
        mc.plt.maskshow(warped_label_t2==100, warped_label_t2==200, warped_label_t2==300, on=image_t1)
        mc.plt.savefig(output_dir / f"{args.exp}_registration_results.png")

    hd_init = args.down_sample * mc.ITKLabelHausdorffDistance(label_t1, label_t2)
    msd_init = args.down_sample * mc.ITKLabelAverageSurfaceDistance(label_t1, label_t2)
    dsc_init = mc.LabelDiceScoreCoefficient(label_t1, label_t2)
    hd_warped = args.down_sample * mc.ITKLabelHausdorffDistance(label_t1, warped_label_t2)
    msd_warped = args.down_sample * mc.ITKLabelAverageSurfaceDistance(label_t1, warped_label_t2)
    dsc_warped = mc.LabelDiceScoreCoefficient(label_t1, warped_label_t2)
    print(f"Validation: HD [{hd_init.mean_std}], MSD [{msd_init.mean_std}], DSC [{dsc_init.mean_std}]")
    print(f"         => HD [{hd_warped.mean_std}], MSD [{msd_warped.mean_std}], DSC [{dsc_warped.mean_std}]")

