python rec_align_reg_matlab.py 0
python rec_align_reg_matlab.py 1
python rec_align_reg_shift_matlab.py 0
python rec_align_reg_shift_matlab.py 1
python cg_reg.py 0
python cg_reg.py 1


# CUDA_VISIBLE_DEVICES=5 python rec_align_reg_matlab.py 1e-10 1e-2 129 &
# CUDA_VISIBLE_DEVICES=6 python rec_align_reg_matlab.py 1e-14 1e-2 129 &
# CUDA_VISIBLE_DEVICES=7 python rec_align_matlab.py 1e-2 129 &
# CUDA_VISIBLE_DEVICES=3 python rec_align_reg_matlab.py 1e-07 1e-2 129 &


# wait


# for k in {16..129..16};do cp rec_align/tmp_168_1e-08/rect$k/r_00072.tiff results/"$k"_1e-08.tiff;done
# for k in {16..129..16};do cp rec_align/tmp_168_1.00001e-08/rect$k/r_00072.tiff results/"$k"_1.00001e-08.tiff;done
# for k in {16..129..16};do cp rec_align/tmp_168/rect$k/r_00072.tiff results/"$k".tiff;done


