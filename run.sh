#!/bin/bash

# Sec 5.1: Reconstruction from Frozen Foundation Models
# -----------------------------------------------------
# Support attacks: (check Python script)
#  rMLE, LM, GLASS, DRAG
# Datasets: (check config/dataset)
#  MS-COCO, FFHQ, ImageNet
# Models: (check config/model)
#  CLIP-ViT-B-16, CLIP-RN50, DINOv2-Base

# python3 run_drag_ldm.py \
#    dataset=mscoco \
#    dataset.target=119,138,725,1044,1703,1919,2111,2591,4111,4497 \
#    model=CLIP-ViT-B-16 \
#    model.split_points=encoder_layer_12 \
#    hydra.sweep.dir=outputs/\${hydra.job.name}/baseline \
#    hydra.sweep.subdir=\${model.checkpoint}/\${dataset.name}.\${dataset.target}/\${model.split_points} \
#    --multirun

# python3 run_drag_ldm.py \
#    dataset=ffhq \
#    dataset.target=337,429,1729,1917,2890,4919,6044,7532,8223,9399 \
#    model=CLIP-ViT-B-16 \
#    model.split_points=encoder_layer_12 \
#    hydra.sweep.dir=outputs/\${hydra.job.name}/baseline \
#    hydra.sweep.subdir=\${model.checkpoint}/\${dataset.name}.\${dataset.target}/\${model.split_points} \
#    --multirun

# python3 run_drag_ldm.py \
#    dataset=imagenet \
#    dataset.target=6091,11341,16904,17849,24681,28026,36044,36293,37807,49165 \
#    model=CLIP-ViT-B-16 \
#    model.split_points=encoder_layer_12 \
#    hydra.sweep.dir=outputs/\${hydra.job.name}/baseline \
#    hydra.sweep.subdir=\${model.checkpoint}/\${dataset.name}.\${dataset.target}/\${model.split_points} \
#    --multirun

# # Sec 5.2: Enhancing DRAG with Inverse Networks
# # ---------------------------------------------
# python3 run_drag_ldm.py \
#    dataset=mscoco \
#    dataset.target=119,138,725,1044,1703,1919,2111,2591,4111,4497 \
#    model=CLIP-ViT-B-16 \
#    model.split_points=encoder_layer_12 \
#    image_prior.latent.init._from_=\${checkpoint_dir}/inverse_network/\${model.checkpoint}/imagenet/\${model.split_points}_public \
#    image_prior.generate_kwargs.strength=0.3 \
#    hydra.sweep.dir=outputs/\${hydra.job.name}++ \
#    hydra.sweep.subdir=\${model.checkpoint}/\${dataset.name}.\${dataset.target}/\${model.split_points} \
#    --multirun

# # Run the Inverse Network only.
# python run_detokenizer.py \
#    +ckpt_dir=checkpoints/inverse_network/openai/clip-vit-base-patch16/imagenet/encoder_layer_12_public \
#    +batch_size=1

# # Sec 5.3.1: Distribution Shift (UCMerced LandUse)
# # ------------------------------------------------
# python3 run_drag_ldm.py \
#    dataset=ucmerced_landuse \
#    dataset.target=1544,1665,1716 \
#    model=CLIP-ViT-B-16 \
#    model.split_points=embeddings,encoder_layer_3,encoder_layer_6,encoder_layer_9,encoder_layer_12 \
#    image_prior.latent.init._from_=\${checkpoint_dir}/inverse_network/\${model.checkpoint}/imagenet/\${model.split_points}_public \
#    image_prior.generate_kwargs.strength=0.3 \
#    hydra.sweep.dir=outputs/\${hydra.job.name}++ \
#    hydra.sweep.subdir=\${model.checkpoint}/\${dataset.name}.\${dataset.target}/\${model.split_points} \
#    --multirun

# # Sec 5.3.2: Distribution Shift (LSUN Bedroom)
# # --------------------------------------------
# python3 run_drag_dm.py \
#    dataset=mscoco \
#    dataset.target=119,138,725,1044,1703,1919,2111,2591,4111,4497 \
#    model=CLIP-ViT-B-16 \
#    model.split_points=encoder_layer_12 \
#    image_prior=lsun_bedroom \
#    hydra.sweep.dir=outputs/\${hydra.job.name} \
#    hydra.sweep.subdir=\${model.checkpoint}/\${dataset.name}.\${dataset.target}/\${model.split_points} \
#    --multirun

# Sec 5.4: Reconstruction from Privacy-Guarded Models
# ---------------------------------------------------
# DISCO:
python3 run_drag_ldm.py \
   dataset=mscoco \
   dataset.target=119,138,725,1044,1703,1919,2111,2591,4111,4497 \
   model=CLIP-ViT-B-16 \
   model.split_points=encoder_layer_12 \
   +defender="disco_\${disco}" \
   +disco=rho-0.95_r-0.1,rho-0.75_r-0.2,rho-0.95_r-0.5 \
   defense.name="channel_pruning" \
   "~defense.kwargs.p" \
   +defense.kwargs.pruner="\${model.checkpoint}" \
   +model._checkpoint=openai/clip-vit-base-patch16 \
   model.checkpoint="\${checkpoint_dir}/disco/\${model._checkpoint}/\${model.split_points}/\${disco}" \
   hydra.sweep.dir=outputs/\${hydra.job.name} \
   hydra.sweep.subdir=\${defender}/\${distance_fn}/\${model._checkpoint}/\${dataset.name}.\${dataset.target}/\${model.split_points} \
   --multirun

# DISCO (Adaptive Attack):
python3 run_drag_ldm.py \
   dataset=mscoco \
   dataset.target=119,138,725,1044,1703,1919,2111,2591,4111,4497 \
   model=CLIP-ViT-B-16 \
   model.split_points=encoder_layer_12 \
   +defender="disco_\${disco}" \
   +disco=rho-0.95_r-0.1,rho-0.75_r-0.2,rho-0.95_r-0.5 \
   defense.name="channel_pruning" \
   "~defense.kwargs.p" \
   +defense.kwargs.pruner="\${model.checkpoint}" \
   +model._checkpoint=openai/clip-vit-base-patch16 \
   model.checkpoint="\${checkpoint_dir}/disco/\${model._checkpoint}/\${model.split_points}/\${disco}" \
   distance_fn=AdaptiveCosineSimilarityLoss \
   hydra.sweep.dir=outputs/\${hydra.job.name} \
   hydra.sweep.subdir=\${defender}/\${distance_fn}/\${model._checkpoint}/\${dataset.name}.\${dataset.target}/\${model.split_points} \
   --multirun

# NoPeek:
python3 run_drag_ldm.py \
   dataset=mscoco \
   dataset.target=119,138,725,1044,1703,1919,2111,2591,4111,4497 \
   model=CLIP-ViT-B-16 \
   model.split_points=encoder_layer_12 \
   +defender="nopeek_lambda-\${nopeek}_dz-cosine" \
   +nopeek=1.0,3.0,5.0 \
   defense.name="channel_pruning" \
   "~defense.kwargs.p" \
   +defense.kwargs.pruner="\${model.checkpoint}" \
   +model._checkpoint=openai/clip-vit-base-patch16 \
   model.checkpoint="\${checkpoint_dir}/nopeek/\${model._checkpoint}/\${model.split_points}/lambda_\${nopeek}_dz_cosine" \
   hydra.sweep.dir=outputs/\${hydra.job.name} \
   hydra.sweep.subdir=\${defender}/\${model._checkpoint}/\${dataset.name}.\${dataset.target}/\${model.split_points} \
   --multirun


# # Sec 5.5: Token Shuffling (and Dropping) Defense
# # -----------------------------------------------
# python3 run_drag_ldm.py \
#    dataset=mscoco \
#    dataset.target=119,138,725,1044,1703,1919,2111,2591,4111,4497 \
#    defense.name=shuffle \
#    defense.kwargs.p=1.0 \
#    adaptive_attack.name=null \
#    model=CLIP-ViT-B-16 \
#    model.split_points=encoder_layer_12 \
#    hydra.sweep.dir=outputs/\${hydra.job.name}/droptoken_r-0.0 \
#    hydra.sweep.subdir=\${model.checkpoint}/\${dataset.name}.\${dataset.target}/\${model.split_points} \
#    --multirun

# python3 run_drag_ldm.py \
#    dataset=mscoco \
#    dataset.target=119,138,725,1044,1703,1919,2111,2591,4111,4497 \
#    defense.name=shuffle \
#    defense.kwargs.p=1.0 \
#    adaptive_attack.name=reorder \
#    model=CLIP-ViT-B-16 \
#    model.split_points=encoder_layer_12 \
#    hydra.sweep.dir=outputs/\${hydra.job.name}/droptoken_r-0.0_reorder \
#    hydra.sweep.subdir=\${model.checkpoint}/\${dataset.name}.\${dataset.target}/\${model.split_points} \
#    --multirun

# python3 run_drag_ldm.py \
#    dataset=mscoco \
#    dataset.target=119 \
#    defense.name=drop_token \
#    defense.kwargs.p=0.5 \
#    adaptive_attack.name=reorder \
#    model=CLIP-ViT-B-16 \
#    model.split_points=encoder_layer_12 \
#    hydra.sweep.dir=outputs/\${hydra.job.name}/droptoken_r-\${defense.kwargs.p}_reorder \
#    hydra.sweep.subdir=\${model.checkpoint}/\${dataset.name}.\${dataset.target}/\${model.split_points} \
#    --multirun


# # Miscellaneous
# # -------------
# # Train an inverse network.
# python train_detokenizer.py \
#    model=DINOv2-Base \
#    model.split_points=encoder_layer_9 \
#    workers=8
