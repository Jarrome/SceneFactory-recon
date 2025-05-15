/*
    Based on https://github.com/wuminye/PCPR

    Yijun Yuan
    Coded at: Feb. 1. 2024
 */


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <torch/extension.h>

void GPU_PCPR(
	torch::Tensor in_points, //(num_points,3)
	torch::Tensor tar_intrinsic, torch::Tensor tar_Pose, 
	float near, float far, float max_splatting_size,
    torch::Tensor out_idx,  //torch::Tensor out_depth, torch::Tensor out_dist2, 
    //torch::Tensor dict_key, torch::Tensor dict_value, torch::Tensor dict_key_sorted, torch::Tensor dict_value_sorted, 
    torch::Tensor ranges);//,
    //torch::Tensor mask_fill, float r); 



