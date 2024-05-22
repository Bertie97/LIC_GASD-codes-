
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
    T1_LIC_sig0 = "--loss LIC --n_batch 1 --n_iter 200 --sig_p 0.", 
    #       => HD [7.225201 ± 1.301809], MSD [0.449450 ± 0.133244], DSC [0.841712 ± 0.069994]
    T1_LIC_sig0_01 = "--loss LIC --n_batch 1 --n_iter 200 --sig_p 0.01", 
    #       => HD [7.652601 ± 2.038758], MSD [0.451383 ± 0.138362], DSC [0.841302 ± 0.071493]
    T1_LIC_sig0_1 = "--loss LIC --n_batch 1 --n_iter 200 --sig_p 0.1", 
    #       => HD [8.110595 ± 2.308115], MSD [0.453210 ± 0.162246], DSC [0.841313 ± 0.078974]
    T1_LIC_sig1 = "--loss LIC --n_batch 1 --n_iter 200 --sig_p 1.", 
    #       => HD [8.052238 ± 3.277528], MSD [0.429991 ± 0.185111], DSC [0.849345 ± 0.085396]
    T1_LIC_sig10 = "--loss LIC --n_batch 1 --n_iter 200 --sig_p 10.", 
    #       => HD [8.399186 ± 2.860154], MSD [0.424723 ± 0.193832], DSC [0.851248 ± 0.088319]
    T1_LIC_noise_sig0 = "--loss LIC --n_batch 1 --n_iter 200 --sig_p 0. --noise 0.05", 
    #       => HD [7.472637 ± 1.496513], MSD [0.500657 ± 0.142104], DSC [0.824546 ± 0.074834]
    T1_LIC_noise_sig0_01 = "--loss LIC --n_batch 1 --n_iter 200 --sig_p 0.01 --noise 0.05", 
    #       => HD [7.909378 ± 2.482459], MSD [0.502289 ± 0.146625], DSC [0.824159 ± 0.076337]
    T1_LIC_noise_sig0_1 = "--loss LIC --n_batch 1 --n_iter 200 --sig_p 0.1 --noise 0.05", 
    #       => HD [7.893785 ± 2.128907], MSD [0.506053 ± 0.157450], DSC [0.823398 ± 0.079510]
    T1_LIC_noise_sig1 = "--loss LIC --n_batch 1 --n_iter 200 --sig_p 1. --noise 0.05", 
    #       => HD [7.801193 ± 2.590768], MSD [0.504140 ± 0.167465], DSC [0.824325 ± 0.083000]
    T1_LIC_noise_sig10 = "--loss LIC --n_batch 1 --n_iter 200 --sig_p 10. --noise 0.05", 
    #       => HD [7.899511 ± 2.529148], MSD [0.504219 ± 0.171173], DSC [0.824271 ± 0.084052]
    T1_CTr_GASD_sig0 = "--loss GASD --n_batch 1 --n_iter 200 --sig_p 0.", 
    #       => HD [15.616043 ± 3.769448], MSD [0.668970 ± 0.138253], DSC [0.771856 ± 0.080289]
    T1_CTr_GASD_sig0_01 = "--loss GASD --n_batch 1 --n_iter 200 --sig_p 0.01", 
    #       => HD [15.607375 ± 2.571915], MSD [0.688870 ± 0.148872], DSC [0.765905 ± 0.084126]
    T1_CTr_GASD_sig0_1 = "--loss GASD --n_batch 1 --n_iter 200 --sig_p 0.1", 
    #       => HD [23.746908 ± 4.257524], MSD [0.811620 ± 0.191497], DSC [0.727414 ± 0.107685]
    T1_CTr_GASD_sig1 = "--loss GASD --n_batch 1 --n_iter 200 --sig_p 1.", 
    #       => HD [38.579048 ± 3.672263], MSD [2.649675 ± 0.647417], DSC [0.460671 ± 0.250568]
    T1_CTr_GASD_sig10 = "--loss GASD --n_batch 1 --n_iter 200 --sig_p 10.", 
    #       => NaN
    T1_CTr_GASD_noise_sig0 = "--loss GASD --n_batch 1 --n_iter 200 --sig_p 0. --noise 0.05", 
    #       => HD [18.324554 ± 0.562146], MSD [0.756759 ± 0.173177], DSC [0.743424 ± 0.095172]
    T1_CTr_GASD_noise_sig0_01 = "--loss GASD --n_batch 1 --n_iter 200 --sig_p 0.01 --noise 0.05", 
    #       => HD [20.811647 ± 2.076759], MSD [0.770123 ± 0.174521], DSC [0.740201 ± 0.096356]
    T1_CTr_GASD_noise_sig0_1 = "--loss GASD --n_batch 1 --n_iter 200 --sig_p 0.1 --noise 0.05", 
    #       => HD [23.632906 ± 4.028473], MSD [0.892752 ± 0.206168], DSC [0.705791 ± 0.114818]
    T1_CTr_GASD_noise_sig1 = "--loss GASD --n_batch 1 --n_iter 200 --sig_p 1. --noise 0.05", 
    #       => HD [38.838150 ± 6.115304], MSD [2.712446 ± 0.598559], DSC [0.443114 ± 0.258703]
    T1_CTr_GASD_noise_sig10 = "--loss GASD --n_batch 1 --n_iter 200 --sig_p 10. --noise 0.05", 
    #       => NaN
    T1_CTr_LIC_sig0 = "--loss CTr_LIC --n_batch 1 --n_iter 200 --sig_p 0.", 
    #       => HD [9.203134 ± 0.976468], MSD [0.809316 ± 0.192489], DSC [0.728895 ± 0.074397]
    T1_CTr_LIC_sig0_01 = "--loss CTr_LIC --n_batch 1 --n_iter 200 --sig_p 0.01", 
    #       => HD [9.282824 ± 0.862421], MSD [0.802647 ± 0.190662], DSC [0.731017 ± 0.073882]
    T1_CTr_LIC_sig0_1 = "--loss CTr_LIC --n_batch 1 --n_iter 200 --sig_p 0.1", 
    #       => HD [9.817883 ± 1.189012], MSD [0.766745 ± 0.188140], DSC [0.744767 ± 0.071447]
    T1_CTr_LIC_sig1 = "--loss CTr_LIC --n_batch 1 --n_iter 200 --sig_p 1.", 
    #       => HD [9.977792 ± 1.632637], MSD [0.630234 ± 0.144007], DSC [0.790534 ± 0.065961]
    T1_CTr_LIC_sig10 = "--loss CTr_LIC --n_batch 1 --n_iter 200 --sig_p 10.", 
    #       => HD [9.465360 ± 1.269038], MSD [0.536634 ± 0.147447], DSC [0.818723 ± 0.074425]
    T1_CTr_LIC_noise_sig0 = "--loss CTr_LIC --n_batch 1 --n_iter 200 --sig_p 0. --noise 0.05", 
    #       => HD [8.115821 ± 1.095532], MSD [0.775146 ± 0.165875], DSC [0.735821 ± 0.076920]
    T1_CTr_LIC_noise_sig0_01 = "--loss CTr_LIC --n_batch 1 --n_iter 200 --sig_p 0.01 --noise 0.05", 
    #       => HD [8.375414 ± 1.666931], MSD [0.770798 ± 0.166803], DSC [0.737408 ± 0.077266]
    T1_CTr_LIC_noise_sig0_1 = "--loss CTr_LIC --n_batch 1 --n_iter 200 --sig_p 0.1 --noise 0.05", 
    #       => HD [8.875209 ± 1.358679], MSD [0.739420 ± 0.167645], DSC [0.747786 ± 0.078828]
    T1_CTr_LIC_noise_sig1 = "--loss CTr_LIC --n_batch 1 --n_iter 200 --sig_p 1. --noise 0.05", 
    #       => HD [8.203186 ± 1.749748], MSD [0.640869 ± 0.154049], DSC [0.780734 ± 0.077946]
    T1_CTr_LIC_noise_sig10 = "--loss CTr_LIC --n_batch 1 --n_iter 200 --sig_p 10. --noise 0.05", 
    #       => HD [8.066013 ± 2.099798], MSD [0.613250 ± 0.161654], DSC [0.788865 ± 0.082733]
    T1_NVI_noise = "--loss NVI --n_batch 1 --n_iter 200 --noise 0.05", 
    #       => HD [7.651484 ± 2.860454], MSD [0.509887 ± 0.111283], DSC [0.820300 ± 0.066467]
    T2_NVI = "--loss NVI --n_batch 1 --n_iter 200", 
    #       => HD [7.995486 ± 3.179354], MSD [0.388325 ± 0.108025], DSC [0.861655 ± 0.061326]
    T2_MI = "--loss MI --n_batch 1 --n_iter 200", 
    #       => HD [26.332586 ± 3.145245], MSD [0.971129 ± 0.398853], DSC [0.734094 ± 0.165662]
    T2_NMI = "--loss NMI --n_batch 1 --n_iter 200", 
    #       => HD [15.916599 ± 2.827151], MSD [0.500597 ± 0.164111], DSC [0.823346 ± 0.087967]
    T2_NCC = "--loss NCC --n_batch 1 --n_iter 200", 
    #       => HD [12.783587 ± 0.932670], MSD [2.329621 ± 0.986703], DSC [0.346581 ± 0.243931]
    T2_NVI_local = "--loss NVI_local --n_batch 1 --n_iter 200", 
    #       => HD [7.610430 ± 2.263193], MSD [0.388207 ± 0.111406], DSC [0.861553 ± 0.062541]
    T2_MI_local = "--loss MI_local --n_batch 1 --n_iter 200", 
    #       => HD [16.594559 ± 0.964850], MSD [0.705605 ± 0.181494], DSC [0.761645 ± 0.088848]
    T2_NCC_local = "--loss NCC_local --n_batch 1 --n_iter 200", 
    #       => HD [7.507532 ± 2.335692], MSD [0.420087 ± 0.122582], DSC [0.850164 ± 0.068227]
    T2_LIC = "--loss LIC --n_batch 1 --n_iter 200", 
    #       => HD [7.225201 ± 1.301809], MSD [0.449450 ± 0.133244], DSC [0.841712 ± 0.069994]
    T2_GASD = "--loss GASD --n_batch 1 --n_iter 200", 
    #       => HD [15.616043 ± 3.769448], MSD [0.668970 ± 0.138253], DSC [0.771856 ± 0.080289]
)

exp_name = args.exp
args.parse_attempt(Experiments[exp_name].split())
args.exp = exp_name

bt.manual_seed(args.seed)
dataset_dir = Path.pardir / "brainweb"
output_dir = Path.curdir.mkdir()

def get_data(k=args.down_sample):
    image_t1 = mc.IMG(dataset_dir / "t1.nii.gz")
    image_t2 = mc.IMG(dataset_dir / "t2.nii.gz")
    image_label = mc.IMG(dataset_dir / "atlasLabel.nii.gz")
    template_t1, template_t2, template_label = (
        image_t1.to_tensor().float().duplicate(args.n_batch, {}) / 150,
        image_t2.to_tensor().float().duplicate(args.n_batch, {}) / 250,
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
    GASD = GASD_loss, 
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
        warped_image_t2 = mc.interpolation(image_t2, trans)
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

