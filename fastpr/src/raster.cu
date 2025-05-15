/*
    Based on https://github.com/wuminye/PCPR

    Yijun Yuan
    Coded at: Feb. 1. 2024
 */


#include "raster.cuh"
#include <stdio.h>

#include <cub/cub.cuh> 
#include <cub/device/device_radix_sort.cuh> 

#include "helper_math.h"


struct Matrix4x4
{
public:
	float4 col[4];
	__device__ __forceinline__
		Matrix4x4()
	{
		col[0] = col[1] = col[2] = col[3] = make_float4(0, 0, 0, 0);
	}
	__device__ __forceinline__
		Matrix4x4(float3 a, float3 b, float3 c, float3 d)
	{
		col[0].x = a.x;
		col[0].y = a.y;
		col[0].z = a.z;
		col[0].w = 0;

		col[1].x = b.x;
		col[1].y = b.y;
		col[1].z = b.z;
		col[1].w = 0;

		col[2].x = c.x;
		col[2].y = c.y;
		col[2].z = c.z;
		col[2].w = 0;

		col[3].x = d.x;
		col[3].y = d.y;
		col[3].z = d.z;
		col[3].w = 1;
	}

	__device__ __forceinline__
		Matrix4x4 transpose() const
	{
		Matrix4x4 res;

		res.col[0].x = col[0].x;
		res.col[0].y = col[1].x;
		res.col[0].z = col[2].x;
		res.col[0].w = col[3].x;

		res.col[1].x = col[0].y;
		res.col[1].y = col[1].y;
		res.col[1].z = col[2].y;
		res.col[1].w = col[3].y;

		res.col[2].x = col[0].z;
		res.col[2].y = col[1].z;
		res.col[2].z = col[2].z;
		res.col[2].w = col[3].z;

		res.col[3].x = 0;
		res.col[3].y = 0;
		res.col[3].z = 0;
		res.col[3].w = 1;
		return res;

	}
	__device__ __forceinline__
		Matrix4x4 inv() const
	{
		Matrix4x4 res;
		res.col[0].x = col[0].x;
		res.col[0].y = col[1].x;
		res.col[0].z = col[2].x;
		res.col[0].w = 0;

		res.col[1].x = col[0].y;
		res.col[1].y = col[1].y;
		res.col[1].z = col[2].y;
		res.col[1].w = 0;

		res.col[2].x = col[0].z;
		res.col[2].y = col[1].z;
		res.col[2].z = col[2].z;
		res.col[2].w = 0;

		res.col[3].x = -dot(col[0], col[3]);
		res.col[3].y = -dot(col[1], col[3]);
		res.col[3].z = -dot(col[2], col[3]);
		res.col[3].w = 1;
		return res;
	}

	__device__ __forceinline__
		static	Matrix4x4 RotateX(float rad)
	{
		Matrix4x4 res;
		res.col[0].x = 1;
		res.col[0].y = 0;
		res.col[0].z = 0;
		res.col[0].w = 0;

		res.col[1].x = 0;
		res.col[1].y = cos(rad);
		res.col[1].z = sin(rad);
		res.col[1].w = 0;

		res.col[2].x = 0;
		res.col[2].y = -sin(rad);
		res.col[2].z = cos(rad);
		res.col[2].w = 0;

		res.col[3].x = 0;
		res.col[3].y = 0;
		res.col[3].z = 0;
		res.col[3].w = 1;
		return res;
	}
};



typedef struct CamPoseNode
{
	//float3 norm, Xaxis, Yaxis, offset;
    float4 r[3];
	__device__ __forceinline__
		Matrix4x4 getRT() const
	{
        Matrix4x4 res;
		res.col[0].x = r[0].x ;
		res.col[0].y = r[1].x ;
		res.col[0].z = r[2].x ;
		res.col[0].w = 0;

		res.col[1].x = r[0].y ;
		res.col[1].y = r[1].y ;
		res.col[1].z = r[2].y ;
		res.col[1].w = 0;

		res.col[2].x = r[0].z ;
		res.col[2].y = r[1].z ;
		res.col[2].z = r[2].z;
		res.col[2].w = 0;

		res.col[3].x = r[0].w ;
		res.col[3].y = r[1].w ;
		res.col[3].z = r[2].w ;
		res.col[3].w = 1;
		return res;
	}

}CamPose;



typedef struct CamIntrinsic
{
	float3 r[3];

	__device__ __forceinline__
		Matrix4x4 getMatrix(float scale = 1.0) const
	{
		Matrix4x4 res;

		res.col[0].x = r[0].x * scale;
		res.col[0].y = r[1].x * scale;
		res.col[0].z = r[2].x * scale;
		res.col[0].w = 0;

		res.col[1].x = r[0].y * scale;
		res.col[1].y = r[1].y * scale;
		res.col[1].z = r[2].y * scale;
		res.col[1].w = 0;

		res.col[2].x = r[0].z * scale;
		res.col[2].y = r[1].z * scale;
		res.col[2].z = r[2].z;
		res.col[2].w = 0;

		res.col[3].x = 0;
		res.col[3].y = 0;
		res.col[3].z = 0;
		res.col[3].w = 1;
		return res;
	}
	__device__ __forceinline__
		float4 PointInverse(float x, float y, float scale = 1.0)
	{
		float xx = (x - r[0].z * scale) / (r[0].x * scale);
		float yy = (y - r[1].z * scale) / (r[1].y * scale);
		return make_float4(xx, yy, 1, 1);
	}

};


namespace math
{
	__device__ __forceinline__
	float4 MatrixMul(const Matrix4x4& mat, float4& x)
	{
		Matrix4x4 res = mat.transpose();
		float4 ans;
		ans.x = dot(res.col[0], x);
		ans.y = dot(res.col[1], x);
		ans.z = dot(res.col[2], x);
		ans.w = dot(res.col[3], x);

		return ans;
	}
}


__global__
void GetTouchTableCUDA(float3 * point_clouds, int num_points,
	CamIntrinsic* tar_intrinsic, CamPose* tar_Pose, int tar_width, int tar_height,
	float near, float far, float max_splatting_size,
	int * touch_table)//,
    //bool * mask_fill, float r)//, float3 * point_xyz)
{
	int ids = blockDim.x * blockIdx.x + threadIdx.x; //  index of point


	if (ids >= num_points) 
		return;

	// Cache camera parameters
	 CamPose _tarcamPose = *tar_Pose;
	 CamIntrinsic _tarcamIntrinsic = *tar_intrinsic;


	float4 p = make_float4(point_clouds[ids], 1.0);
    //printf("p %f, %f, %f, %f \n", p.x, p.y, p.z, p.w);

    // the input is inverse pose
	Matrix4x4 camT = _tarcamPose.getRT();
	camT = camT.inv();
	float4 camp = math::MatrixMul(camT, p);




	float tdepth = camp.z;

	if (tdepth < 0.1)
		return;

	//float cosine = camp.z/sqrt(camp.x*camp.x+camp.y*camp.y+camp.z*camp.z+0.000001);


    Matrix4x4 K = _tarcamIntrinsic.getMatrix();

	camp = camp / camp.z;

    //printf("%d before %f, %f, %f, %f \n", ids, camp.x, camp.y, camp.z, camp.w);
	camp = math::MatrixMul(K, camp);

    //printf("%d after %f, %f, %f, %f \n", ids, camp.x, camp.y, camp.z, camp.w);

	//camp = camp / camp.w;



	// splatting radius
    /*
	float rate = (tdepth - near) / (far - near);
	rate = 1.0 - rate;
	rate = max(rate, 0.0);
	rate = min(rate, 1.0);
    */

    // spatial radius = 0.005/2
    // fx * 0.005/2 is the radius in pixel at 1m
    // fx/2 * 0.005/2 is at 2m
    // fx*2 * 0.005/2 is at .5m
    // the cosine doesnot affect the spatial space of each pixel
    // because pixel is with rectangule, the 1 pixel is sqrt(2)
	float radius = 0.005/2 * K.col[0].x/tdepth;// * 1.41421356237;// * r; 


	int xstart = round(camp.x - radius );
	int ystart = round(camp.y - radius );
	int xend = round(camp.x + radius );
	int yend = round(camp.y + radius );
 
    touch_table[ids] = 0;
	if (xstart >= tar_width || ystart >= tar_height || xend<0 || yend<0)
		return;
    /*
    point_xyz[ids].x = camp.x;
    point_xyz[ids].y = camp.y;
    point_xyz[ids].z = tdepth;
    */

    // assigning
	for (int xx = xstart; xx <= xend; ++xx)
	{
		for (int yy = ystart; yy <= yend; ++yy)
		{
            //if (!mask_fill[yy*tar_width+xx])
             //   continue;

			if (xx < 0 || xx >= tar_width || yy < 0 || yy >= tar_height)
                continue;
            touch_table[ids] += 1;
		}
	}

}

__global__
void PointProcessCUDA(float3 * point_clouds, int num_points,
	CamIntrinsic* tar_intrinsic, CamPose* tar_Pose, int tar_width, int tar_height,
	float near, float far, float max_splatting_size,
	int * touch_offset, int * dict_key, int * dict_value)//,
    //bool * mask_fill,
    //float r)//, float3 * point_xyz)
{
	int ids = blockDim.x * blockIdx.x + threadIdx.x; //  index of point


	if (ids >= num_points) 
		return;

    int num_pix, offset;
    if (ids == 0){
        offset = 0;
        num_pix = touch_offset[0];
    }
    else{
        offset = touch_offset[ids-1];
        num_pix = touch_offset[ids] - touch_offset[ids-1];
    }

    if (num_pix <= 0)
        return;

	// Cache camera parameters
	 CamPose _tarcamPose = *tar_Pose;
	 CamIntrinsic _tarcamIntrinsic = *tar_intrinsic;


	float4 p = make_float4(point_clouds[ids], 1.0);
    //printf("p %f, %f, %f, %f \n", p.x, p.y, p.z, p.w);

    // the input is inverse pose
	Matrix4x4 camT = _tarcamPose.getRT();
	camT = camT.inv();
	float4 camp = math::MatrixMul(camT, p);




	float tdepth = camp.z;

	if (tdepth < 0.1)
		return;

	//float cosine = camp.z/sqrt(camp.x*camp.x+camp.y*camp.y+camp.z*camp.z+0.000001);


    Matrix4x4 K = _tarcamIntrinsic.getMatrix();

	camp = camp / camp.z;

    //printf("%d before %f, %f, %f, %f \n", ids, camp.x, camp.y, camp.z, camp.w);
	camp = math::MatrixMul(K, camp);
    //printf("%d after %f, %f, %f, %f \n", ids, camp.x, camp.y, camp.z, camp.w);

	//camp = camp / camp.w;


	// splatting radius
    /*
	float rate = (tdepth - near) / (far - near);
	rate = 1.0 - rate;
	rate = max(rate, 0.0);
	rate = min(rate, 1.0);
    */

    // spatial radius = 0.005/2
    // fx * 0.005/2 is the radius in pixel at 1m, 
    // fx/2 * 0.005/2 is at 2m
    // fx*2 * 0.005/2 is at .5m
    // the cosine doesnot affect the spatial space of each pixel
    // because pixel is with rectangule, the 1 pixel is sqrt(2)
	float radius = 0.005/2 * K.col[0].x/tdepth; // * 1.41421356237;// * r; 


	int xstart = round(camp.x - radius );
	int ystart = round(camp.y - radius );
	int xend = round(camp.x + radius );
	int yend = round(camp.y + radius );
 
	if (xstart >= tar_width || ystart >= tar_height || xend<0 || yend<0)
		return;
    /*
    point_xyz[ids].x = camp.x;
    point_xyz[ids].y = camp.y;
    point_xyz[ids].z = tdepth;
    */

    // assigning
    int offset_i = 0;
	for (int xx = xstart; xx <= xend; ++xx)
	{
		for (int yy = ystart; yy <= yend; ++yy)
		{
            //if (!mask_fill[yy*tar_width+xx])
            //    continue;
			if (xx < 0 || xx >= tar_width || yy < 0 || yy >= tar_height)
                continue;

			int ind = yy * tar_width + xx ;
            
            dict_key[offset+offset_i] = ind;
            dict_value[offset+offset_i] = ids;

            offset_i += 1;
		}
	}

}

__global__ void IdentifyPixRange(int L, int N_pix, int * dict_keys_sorted, int * ranges)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x; //  index of key

    if (idx >= L)
        return;

    // Read pix ID from key. Update start/end of tile range if at limit.
    int key = dict_keys_sorted[idx];
    if (key >= N_pix)
        return;

    int currpix = key ;//>> 32;
    if (idx == 0)
        ranges[currpix*2] = 0;
    else
    {
        int prevpix = dict_keys_sorted[idx - 1];// >> 32;
        if (currpix != prevpix)
        {
            ranges[prevpix*2+1] = idx;
            ranges[currpix*2] = idx;
        }
    }
}

__global__
void PixProcessCUDA(float3 * point_clouds, int num_points,
	CamIntrinsic* tar_intrinsic, CamPose* tar_Pose, int tar_width, int tar_height,
	float near, float far, float max_splatting_size,
	int * dict_value, int * ranges, int * out_index) //, float * out_depth, float * out_dist2)//,
    //float3 * point_xyz)
{

    int pix_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pix_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (pix_x >= tar_width || pix_y >= tar_height)
		return;

    int idx = pix_x + pix_y * tar_width;    

    // get range
    int range_start= ranges[idx*2];
    int range_end = ranges[idx*2+1];

	// Cache camera parameters
	 CamPose _tarcamPose = *tar_Pose;
	 CamIntrinsic _tarcamIntrinsic = *tar_intrinsic;

  int k = 0;
  float quart_HW = tar_height * tar_width / 4;
  for (int ii = range_start; ii < range_end; ii++){ 
    if (k==8)
        return;
    int ids = dict_value[ii];
    out_index[idx*8+k] = ids;
    //out_depth[idx*8+k] = point_xyz[ids].z;
    //out_dist2[idx*8+k] = ((pix_x - point_xyz[ids].x)*(pix_x - point_xyz[ids].x)  + (pix_y - point_xyz[ids].y) * (pix_y - point_xyz[ids].y) ) / quart_HW;
    k++;
  }

    /*
    int ids = dict_value[ii];
	float4 p = make_float4(point_clouds[ids], 1.0);
	Matrix4x4 camT = _tarcamPose.getRT();
	camT = camT.inv();
	float4 camp = math::MatrixMul(camT, p);
	float tdepth = camp.z;

	if (tdepth < 0)
        continue;

	float cosine = camp.z/sqrt(camp.x*camp.x+camp.y*camp.y+camp.z*camp.z+0.000001);

	camp = math::MatrixMul(_tarcamIntrinsic.getMatrix(), camp);

	camp = camp / camp.w;
	camp = camp / camp.z;



	// splatting radius

	float rate = (tdepth - near) / (far - near);
	rate = 1.0 - rate;
	rate = max(rate, 0.0);
	rate = min(rate, 1.0);
	

	float radius = (max_splatting_size * rate) / cosine;

			int ind = yy * tar_width + xx ;
            
            dict_key[offset+offset_i] = ind;
            dict_value[offset+offset_i] = ids;
            
            offset_i += 1;
		}
	}
    */

}



void GPU_PCPR(
	torch::Tensor in_points, //(num_points,3)
	torch::Tensor tar_intrinsic, torch::Tensor tar_Pose, 
	float near, float far, float max_splatting_size,
	torch::Tensor out_index, //torch::Tensor out_depth, torch::Tensor out_dist2, 
    //torch::Tensor dict_key, torch::Tensor dict_value, torch::Tensor dict_key_sorted, torch::Tensor dict_value_sorted, 
    torch::Tensor ranges)//,
    //torch::Tensor mask_fill, float r) 
{
    bool debug = true;
	const auto num_points = in_points.size(0);

	dim3 dimBlock(256,1);
	dim3 dimGrid(num_points / 256 + 1, 1);

	int tar_height = out_index.size(0);
	int tar_width = out_index.size(1);
    /*
    float3 * point_xyz; // xy is pix xy, z is depth
    CHECK_CUDA(cudaMalloc(&point_xyz, num_points*sizeof(float3)),debug)
    //CHECK_CUDA(cudaMemset(point_xyz, 0, num_points*sizeof(float3)),debug)
    */

    // traverse point to get touch table
    int * touch_table;
    CHECK_CUDA(cudaMalloc(&touch_table, num_points*sizeof(int)),debug);
    //CHECK_CUDA(cudaMemset(touch_table, 0, num_points*sizeof(int)), debug);

    GetTouchTableCUDA << <dimGrid, dimBlock >> > (
		(float3*)in_points.data<float>(), num_points,
		(CamIntrinsic*)tar_intrinsic.data<float>(),(CamPose*)tar_Pose.data<float>(), tar_width, tar_height,
		near, far, max_splatting_size,
		touch_table//,
        //mask_fill.data<bool>(),r
        );



    // allocate dict with touch table
    int * touch_offset;
    CHECK_CUDA(cudaMalloc(&touch_offset, num_points*sizeof(int)),debug);
   
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, touch_table, touch_offset, num_points);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, touch_table, touch_offset, num_points);

    int * touch_offset_host;
    touch_offset_host = (int *)malloc(num_points*sizeof(int));
    cudaMemcpy(touch_offset_host, touch_offset, num_points*sizeof(int), cudaMemcpyDeviceToHost);

    int len_dict = touch_offset_host[num_points-1];

    cudaFree(d_temp_storage);

    int * dict_key;
    int * dict_value;
    cudaMalloc(&dict_key, len_dict*sizeof(int));
    cudaMalloc(&dict_value, len_dict*sizeof(int));

    // build key value
	PointProcessCUDA << <dimGrid, dimBlock >> > (
		(float3*)in_points.data<float>(), num_points,
		(CamIntrinsic*)tar_intrinsic.data<float>(),(CamPose*)tar_Pose.data<float>(), tar_width, tar_height,
		near, far, max_splatting_size,
        touch_offset, dict_key, dict_value//,
        //mask_fill.data<bool>(),r
        );
    // sort dict by keys,values
    int * dict_key_sorted;
    int * dict_value_sorted;
    cudaMalloc(&dict_key_sorted, len_dict*sizeof(int));
    cudaMalloc(&dict_value_sorted, len_dict*sizeof(int));

    d_temp_storage = NULL;
    temp_storage_bytes = 0;
    CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
        d_temp_storage,
        temp_storage_bytes,
        dict_key, dict_key_sorted,                             
        dict_value, dict_value_sorted,
        len_dict),debug)
    CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes),debug)
    CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
        d_temp_storage,
        temp_storage_bytes,
        dict_key, dict_key_sorted,                             
        dict_value, dict_value_sorted,
        len_dict),debug)

    // build range for each pixel
    IdentifyPixRange<< <(len_dict + 255) / 256, 256 >> >(
        len_dict, tar_height*tar_width, dict_key_sorted, ranges.data<int>());

    // splatting
    dim3 pix_block(16,16,1);
    dim3 pix_grid((tar_width + 16 - 1) / 16, (tar_height + 16 - 1) / 16, 1);
	PixProcessCUDA << <pix_grid, pix_block >> > (
		(float3*)in_points.data<float>(), num_points,
		(CamIntrinsic*)tar_intrinsic.data<float>(),(CamPose*)tar_Pose.data<float>(), tar_width, tar_height,
		near, far, max_splatting_size,
		dict_value_sorted, ranges.data<int>(), 
        out_index.data<int>()// out_depth.data<float>(), out_dist2.data<float>()
        );

    
    cudaFree(touch_table);
    cudaFree(touch_offset);
    free(touch_offset_host);
    cudaFree(d_temp_storage);

    cudaFree(dict_key);
    cudaFree(dict_value);

    cudaFree(dict_key_sorted);
    cudaFree(dict_value_sorted);
}


