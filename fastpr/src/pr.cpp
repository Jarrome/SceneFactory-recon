/*
    Based on https://github.com/wuminye/PCPR

    Yijun Yuan
    Coded at: Feb. 1. 2024
 */


#include <torch/extension.h>
#include <vector>
#include "raster.cuh"
#include <iostream>


// CUDA forward declarations
/*
std::vector<torch::Tensor> pcpr_cuda_forward(
    torch::Tensor in_points, //(num_points,3)
    torch::Tensor tar_intrinsic, torch::Tensor tar_Pose,
    torch::Tensor out_depth, torch::Tensor out_index, torch::Tensor out_dist, torch::Tensor cover_count, torch::Tensor im_max_idx,// (tar_heigh ,tar_width)
    float near, float far, float max_splatting_size
    );
*/

// C++ interface

#define CHECK_CPU(x) AT_ASSERTM(!x.type().is_cuda(), #x " must be a CPU tensor")
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) AT_ASSERTM(x.type().scalarType()==torch::ScalarType::Float, #x " must be a float tensor")
#define CHECK_INT(x) AT_ASSERTM(x.type().scalarType()==torch::ScalarType::Int, #x " must be a Int tensor")
#define CHECK_SHORT(x) AT_ASSERTM(x.type().scalarType()==torch::ScalarType::Short, #x " must be a Int tensor")
#define CHECK_UCHAR(x) AT_ASSERTM(x.type().scalarType()==torch::ScalarType::Byte, #x " must be a Int tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);

std::vector<torch::Tensor> pcpr_cuda_forward(
    torch::Tensor in_points, //(num_points,3)
    torch::Tensor tar_intrinsic, torch::Tensor tar_Pose,
    float near, float far, float max_splatting_size,
    torch::Tensor out_index,  //torch::Tensor out_depth, torch::Tensor out_dist2, 
    //torch::Tensor dict_key, torch::Tensor dict_value, torch::Tensor dict_key_sorted, torch::Tensor dict_value_sorted,
    torch::Tensor ranges)//,
    //torch::Tensor mask_fill, float r) 
{
  CHECK_INPUT(in_points); CHECK_FLOAT(in_points);
  CHECK_INPUT(tar_intrinsic); CHECK_FLOAT(tar_intrinsic);
  CHECK_INPUT(tar_Pose); CHECK_FLOAT(tar_Pose);
  CHECK_INPUT(out_index); CHECK_INT(out_index);

 // AT_ASSERTM(out_depth.size(0)== out_index.size(0), "out_depth and out_index must be the same size");
  //AT_ASSERTM(out_depth.size(1)== out_index.size(1), "out_depth and out_index must be the same size");

  GPU_PCPR(
	in_points, //(num_points,3)
	tar_intrinsic, tar_Pose, 
	near, far, max_splatting_size,
    out_index, //out_depth, out_dist2, 
    //dict_key, dict_value, dict_key_sorted, dict_value_sorted,
    ranges);//,
    //mask_fill, r); 



  return {out_index, ranges};//dict_key, dict_value, dict_key_sorted, dict_value_sorted, ranges};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &pcpr_cuda_forward, "PCPR forward (CUDA)");
}

