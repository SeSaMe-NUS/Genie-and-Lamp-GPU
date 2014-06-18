#ifndef SAVE_RESULT_INC_
#define SAVE_RESULT_INC_

static __global__ void save_result(Bucket* bucket, int* stat_index, int* end_index, int* last_bucket_index, int* thread_start_index, int* thread_end_index, int* result)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int threadId = bid * blockDim.x + tid;

	int blk_start_index = stat_index[bid];
	int blk_end_index = end_index[bid];

	int round = (blk_end_index-blk_start_index)/blockDim.x + 1;
	int offset = blockDim.x;

	int index;
	int position = thread_start_index[threadId];

	for(int i=0; i<round; i++)
	{
		index = (blk_start_index+tid)+i*offset;
		if(index < blk_end_index)
		{
			if(bucket[index].b_index < last_bucket_index[bid] && position < thread_end_index[threadId])
			{
				result[position] = bucket[index].a_index;
				position++;
			}
		}
	}
}

static __global__ void save_result(Bucket* bucket, int* end_index, int* last_bucket_index, float* min, float* max, int* thread_start_index, int* thread_end_index, int* result)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int threadId = bid * blockDim.x + tid;

	int blk_start_index = (bid==0) ? 0 : end_index[bid-1];
	int blk_end_index = end_index[bid];

	int round = (blk_end_index-blk_start_index)/blockDim.x + 1;
	int offset = blockDim.x;

	int index;
	int position = thread_start_index[threadId];

	for(int i=0; i<round; i++)
	{
		index = (blk_start_index+tid)+i*offset;
		if(index < blk_end_index)
		{
			if(bucket[index].b_index < last_bucket_index[bid] && position < thread_end_index[threadId] && !BUCKET_TOPK_EQUAL(min[bid], max[bid]))
			{
				result[position] = bucket[index].a_index;
				position++;
			}
			if(bucket[index].b_index == last_bucket_index[bid] && position < thread_end_index[threadId] && BUCKET_TOPK_EQUAL(min[bid], max[bid]))
			{
				result[position] = bucket[index].a_index;
				position++;
			}
		}
	}
}

static __global__ void save_miss_array(Bucket* bucket, int* end_index, int* last_bucket_index, int* thread_start_index, int* thread_end_index, int* result)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int threadId = bid * blockDim.x + tid;

	int blk_start_index = (bid==0) ? 0 : end_index[bid-1];
	int blk_end_index = end_index[bid];

	int round = (blk_end_index-blk_start_index)/blockDim.x + 1;
	int offset = blockDim.x;

	int index;
	int position = thread_start_index[threadId];

	for(int i=0; i<round; i++)
	{
		index = (blk_start_index+tid)+i*offset;
		if(index < blk_end_index)
		{
			if(bucket[index].b_index == last_bucket_index[bid] && position < thread_end_index[threadId])
			{
				result[position] = bucket[index].a_index;
				position++;
			}
		}
	}
}

static __global__ void save_miss_array(Bucket* bucket, int* start_index, int* end_index, int* last_bucket_index, int* thread_start_index, int* thread_end_index, int* result)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int threadId = bid * blockDim.x + tid;

	int blk_start_index = start_index[bid];
	int blk_end_index = end_index[bid];

	int round = (blk_end_index-blk_start_index)/blockDim.x + 1;
	int offset = blockDim.x;

	int index;
	int position = thread_start_index[threadId];

	for(int i=0; i<round; i++)
	{
		index = (blk_start_index+tid)+i*offset;
		if(index < blk_end_index)
		{
			if(bucket[index].b_index == last_bucket_index[bid] && position < thread_end_index[threadId])
			{
				result[position] = bucket[index].a_index;
				position++;
			}
		}
	}
}

static __global__ void update_result_last_position(int* result_last_position, int* thread_end_index)
{
	int bid = blockIdx.x;
	result_last_position[bid] = thread_end_index[(bid+1)*blockDim.x-1];
}

template<typename Data>
static __global__ void shift_result(Data* data, int* start_index, int* result, int* end_index_of_result)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int start = start_index[bid];
	int result_start = (bid==0) ? 0 : end_index_of_result[bid-1];
	int result_end = end_index_of_result[bid];

	Data newDatak, oldDatak;

	if(tid+result_start < result_end)
	{
		newDatak = data[result[tid+result_start]];
		oldDatak = data[tid+start];
	}
	__syncthreads();
	if(tid+result_start < result_end)
	{
		data[result[tid+result_start]] = oldDatak;
		data[tid+start] = newDatak;
	}
}

template<typename Data>
static __global__ void shift_result_for(Data* data, int* start_index, int* result, int* end_index_of_result)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int start = start_index[bid];
	int result_start = (bid==0) ? 0 : end_index_of_result[bid-1];
	int result_end = end_index_of_result[bid];

	if(tid==0)
	{
		for(int i=result_end-1; i>=result_start; i--)
			for(int j=result_start; j<i; j++)
				if(result[j] >  result[j+1])
				{
					int temp = result[j];
					result[j] = result[j+1];
					result[j+1] = temp;
				}
		Data t;
		for(int i=result_start; i<result_end; i++)
		{
			t = data[i-result_start+start];
			data[i-result_start+start] = data[result[i]];
			data[result[i]] = t;
		}
	}
}

#endif
