python main.py --ae --train --epoch 200 --sample_dir samples/all_vox256_img0_16 --sample_vox_size 16 --checkpoint_dir ckpt_ShapeTalkClasses_pub_scaled --splits /orion/u/ianhuang/shapetalk_retraining/unary_splits.csv --data_dir vox_preprocessing3
python main.py --ae --train --epoch 200 --sample_dir samples/all_vox256_img0_32 --sample_vox_size 32 --checkpoint_dir ckpt_ShapeTalkClasses_pub_scaled --splits /orion/u/ianhuang/shapetalk_retraining/unary_splits.csv --data_dir vox_preprocessing3
python main.py --ae --train --epoch 400 --sample_dir samples/all_vox256_img0_64 --sample_vox_size 64 --checkpoint_dir ckpt_ShapeTalkClasses_pub_scaled --splits /orion/u/ianhuang/shapetalk_retraining/unary_splits.csv --data_dir vox_preprocessing3
# python main.py --ae --getz
