
try: from pycamia import info_manager
except ImportError: print("Warning: pycamia not loaded. ")

__info__ = info_manager(
    project = "Intensity Distribution",
    package = "Experiment 3",
    author = "Anonymous", 
    create = "2024-03",
    fileinfo = "Experiment 3: comparison on real data: MS-CMR. ",
    requires = ["pycamia", "batorch", "micomputing"]
)

import os, sys, re, random
import batorch as bt
import micomputing as mc
from datetime import datetime
from pycamia import (
    args, scope, Jump, 
    Path, logging, cat_generator
)

sys.path.append(Path.pardir.abs)
from color_transfer import color_transfer, LocalIntensityCorrelation

args.seed = 1234
args.loss = "NVI"
args.n_dim = 3
args.n_batch = 2
args.n_epoch = 100
args.n_epoch_save = 5
args.n_affine_block = 3
args.offset_scale = 5
args.sig_p = 0.
args.noise = 0.
args.down_sample = 2
args.exp = "T3_LIC"

Experiments = dict(
    # initial: HD [10.337495 ± 2.533981], MSD [1.464208 ± 0.486360], DSC [0.668198 ± 0.087080]
    T3_LIC = "--loss LIC --n_batch 1 --n_epoch 100", 
    #       => HD [15.533257 ± 5.241134], MSD [1.473094 ± 0.508876], DSC [0.705433 ± 0.053999]
    T3_NVI = "--loss NVI --n_batch 1 --n_epoch 100", 
    #       => HD [8.366378 ± 2.905711], MSD [1.199061 ± 0.575339], DSC [0.696201 ± 0.114326]  
    T3_MI = "--loss MI --n_batch 1 --n_epoch 100", 
    #       => HD [10.136765 ± 1.474427], MSD [1.499573 ± 0.277208], DSC [0.683085 ± 0.043499]
    T3_MI_local = "--loss MI_local --n_batch 1 --n_epoch 100", 
    #       => HD [8.985089 ± 1.551091], MSD [1.162352 ± 0.353054], DSC [0.704623 ± 0.089992]
    T3_NCC = "--loss NCC --n_batch 1 --n_epoch 100", 
    #       => HD [9.246312 ± 2.135172], MSD [1.284004 ± 0.400621], DSC [0.699023 ± 0.073490]
    T3_NCC_local = "--loss NCC_local --n_batch 1 --n_epoch 100", 
    #       => HD [9.382143 ± 1.431658], MSD [1.311132 ± 0.305074], DSC [0.673166 ± 0.071691]
)

exp_name = args.exp
args.parse_attempt(Experiments[args.exp].split())
args.exp = exp_name

def GASD_loss(i1, i2):
    kernel = bt.gaussian_kernel(n_dim = args.n_dim, kernel_size = 3).unsqueeze([]).duplicate(args.n_batch, {})
    mean = lambda x: eval("bt.nn.functional.conv%dd"%args.n_dim)(x.unsqueeze([1]), kernel, padding = 1).squeeze([])
    translated_i1 = color_transfer(i1, i2, args.sig_p)
    return - mean((i2 - translated_i1) ** 2).mean()

def CTr_LIC_loss(i1, i2):
    kernel = bt.gaussian_kernel(n_dim = args.n_dim, kernel_size = 3).unsqueeze([]).duplicate(args.n_batch, {})
    mean = lambda x: eval("bt.nn.functional.conv%dd"%args.n_dim)(x.unsqueeze([1]), kernel, padding = 1).squeeze([])
    translated_i1 = color_transfer(i1, i2, args.sig_p)
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

random.seed(args.seed)
bt.manual_seed(args.seed)
dataset_dir = Path.pardir / "MS-CMR-REG"
output_dir = Path.curdir.mkdir()

@mc.Dataset(dataset_dir, batch_pattern=[
    mc.Key(subject_id=..., modality='C0', affine_info=False), 
    mc.Key(subject_id=..., modality='C0_label', affine_info=False), 
    mc.Key(subject_id=..., modality='C0', affine_info=True), 
    mc.Key(subject_id=..., modality='LGE', affine_info=False), 
    mc.Key(subject_id=..., modality='LGE_label', affine_info=False), 
    mc.Key(subject_id=..., modality='LGE', affine_info=True)
])
def MS_CMR(path):
    if not path | 'nii.gz': return
    pid, mod = path.name.split('_', 1)
    img = mc.IMG(path-Path.curdir)
    if mod in ('C0', 'LGE0'):
        data = mc.Normalize()(bt.to_device(img.to_tensor()).float().duplicate(args.n_batch, {}))
        data = (data + args.noise * bt.randn_like(data)).clamp(0, 1)
    elif mod in ('C0_label', 'LGE0_label'):
        data = bt.to_device(img.to_tensor()).duplicate(args.n_batch, {})
    else: return None
    mod = mod.replace('LGE0', 'LGE')
    data = data[..., *(slice(None, None, args.down_sample),) * args.n_dim]
    return [(mc.Key(subject_id=pid, modality=mod, affine_info=False), data), 
            (mc.Key(subject_id=pid, modality=mod, affine_info=True), img.affine)]

register = mc.U_Net(
    dimension = args.n_dim, 
    in_channels = 2, 
    out_channels = args.n_dim, 
    block_channels = 8, 
    # bottleneck_out_channels = args.n_dim * (args.n_dim + 1), 
    with_softmax = False, 
    initializer = "normal(0, 1e-3)"
).to(bt.get_device().device)

# affiner = mc.CNN(
#     dimension = args.n_dim, 
#     blocks = args.n_affine_block, 
#     in_channels = 2, 
#     out_elements = args.n_dim * (args.n_dim + 1), 
#     with_softmax = False, 
#     initializer = "normal(0, 1e-4)"
# ).to(bt.get_device().device)

first_batch = True
optimizer = bt.Optimization(bt.optim.Adam, register.parameters(), lr=lambda i: 1e-4 * 0.4 ** (i / args.n_epoch / 30))

with logging(log_path=output_dir/f"{Path(__file__).name}_{args.exp}_{datetime.now().strftime('%Y%m%d%H%M%S')}.log"):
    for i_epoch in range(args.n_epoch):
        i_iter = 0
        for batch in MS_CMR.training_batches(args.n_batch, shuffle=False):
            image_c0, label_c0, affine_c0, image_lge, label_lge, affine_lge = batch
            
            attention_image_c0 = (image_c0 * 0.2 * (1 + 4 * mc.blur((label_c0 > 0).float())))
            attention_image_lge = (image_lge * 0.2 * (1 + 4 * mc.blur((label_lge > 0).float())))

            # with scope("plot init images"), Jump(not first_batch):
            #     mc.plt.figure()
            #     mc.plt.subplots(4)
            #     mc.plt.imsshow(image_c0, image_lge, label_c0, label_lge, as_orient=mc.affine2orient(affine_c0[0]))
            #     mc.plt.savefig(output_dir / "transformed_images.png")
            
            common_shape = tuple(max(x, y) for x, y in zip(image_c0.space, image_lge.space))
            inputs = bt.stack(bt.crop_as(image_c0, common_shape), bt.crop_as(image_lge, common_shape), [])
            offsets = args.offset_scale * register(inputs)
            # affine = bt.cat(1e-1 * affiner(inputs).split_dim([], [args.n_dim, args.n_dim + 1]) + bt.cat(bt.eye([args.n_dim]), bt.zeros([args.n_dim, 1]), 1), bt.one_hot(-1, args.n_dim + 1).view([1, args.n_dim+1]), [0])
            trans = mc.DDF(offsets, domain='physical-space', use_implementation='image-space').between_spaces(target=affine_c0[0], source=affine_lge[0])
            
            warped_image_lge = trans(attention_image_lge, to_shape=image_c0.space, domain='non-spatial')
            warped_label_lge = trans(label_lge, to_shape=image_c0.space, domain='non-spatial', method='Nearest')
            
            similarity = sim_func[args.loss](attention_image_c0, warped_image_lge)
            loss = - similarity.mean() + 1 * mc.bending(offsets).mean()
            opt = optimizer.new_iter()
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            dsc = mc.LabelDiceScoreCoefficient(label_c0.long(), warped_label_lge.long())
            print(f"Epoch {i_epoch+1}; Iteration {i_iter+1}: similarity = {similarity.mean().item():.2e}, loss = {loss.item():.2e}, dice = {dsc.mean().item():.2e}")
            i_iter += 1

            # with scope("plot registration results"), Jump(not first_batch):
            #     mc.plt.figure()
            #     mc.plt.subplots(2)
            #     mc.plt.maskshow(warped_label_lge==200, warped_label_lge==500, warped_label_lge==600, on=image_lge)
            #     mc.plt.maskshow(warped_label_lge==200, warped_label_lge==500, warped_label_lge==600, on=image_c0)
            #     mc.plt.savefig(output_dir / "registration_results.png")
            
            first_batch = False

        print(f"Validation {i_epoch+1}")
        collect_init = mc.LossCollector()
        collect = mc.LossCollector()
        i_case = 0
        for batch in MS_CMR.validation_batches(args.n_batch, shuffle=False):
            image_c0, label_c0, affine_c0, image_lge, label_lge, affine_lge = batch
            common_shape = tuple(max(x, y) for x, y in zip(image_c0.space, image_lge.space))
            inputs = bt.stack(bt.crop_as(image_c0, common_shape), bt.crop_as(image_lge, common_shape), [])
            offsets = args.offset_scale * register(inputs)
            id_trans = mc.Identity().between_spaces(target=affine_c0[0], source=affine_lge[0])
            trans = mc.DDF(offsets, domain='physical-space', use_implementation='image-space').between_spaces(target=affine_c0[0], source=affine_lge[0])
            
            init_label_lge = id_trans(label_lge, to_shape=image_c0.space, domain='non-spatial', method='Nearest')
            warped_label_lge = trans(label_lge, to_shape=image_c0.space, domain='non-spatial', method='Nearest')
            hd_init = args.down_sample * mc.ITKLabelHausdorffDistance(label_c0, init_label_lge)
            msd_init = args.down_sample * mc.ITKLabelAverageSurfaceDistance(label_c0, init_label_lge)
            dsc_init = mc.LabelDiceScoreCoefficient(label_c0, init_label_lge)
            hd_warped = args.down_sample * mc.ITKLabelHausdorffDistance(label_c0, warped_label_lge)
            msd_warped = args.down_sample * mc.ITKLabelAverageSurfaceDistance(label_c0, warped_label_lge)
            dsc_warped = mc.LabelDiceScoreCoefficient(label_c0, warped_label_lge)
            heading = f"Case {i_case+1}"
            print(f"{heading}: HD [{hd_init.mean_std}], MSD [{msd_init.mean_std}], DSC [{dsc_init.mean_std}]")
            print(f"{' ' * (len(heading) - 1)}=> HD [{hd_warped.mean_std}], MSD [{msd_warped.mean_std}], DSC [{dsc_warped.mean_std}]")
            collect_init(HD=hd_init.mean(), MSD=msd_init.mean(), DSC=dsc_init.mean())
            collect(HD=hd_warped.mean(), MSD=msd_warped.mean(), DSC=dsc_warped.mean())
            i_case += 1
        heading = f"Validation {i_epoch+1}"
        print(f"{heading}: {collect_init}")
        print(f"{' ' * (len(heading) - 1)}=> {collect}")

        if (i_epoch + 1) % args.n_epoch_save == 0:
            bt.save(register.state_dict(), output_dir / f"{args.exp}_state_dict_epoch{i_epoch+1:03d}.ckpt")

