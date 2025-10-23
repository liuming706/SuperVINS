#include "extractor_matcher_dpl.h"
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif
Extractor_DPL::Extractor_DPL(unsigned int _num_threads)
    : num_threads(_num_threads) {}

void Extractor_DPL::initialize(const std::string &extractorPath,
                               int extractor_type_) {
  extractor_type = extractor_type_;
  env = Ort::Env(ORT_LOGGING_LEVEL_WARNING,
                 "LightGlueDecoupleOnnxRunner Extractor");
  session_options = Ort::SessionOptions();
  session_options.SetInterOpNumThreads(std::thread::hardware_concurrency());
  session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);

  OrtCUDAProviderOptions cuda_options{};
  cuda_options.device_id = 0;
  cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
  cuda_options.gpu_mem_limit = 0;
  cuda_options.arena_extend_strategy = 1;  // 设置GPU内存管理中的Arena扩展策略
  cuda_options.do_copy_in_default_stream = 1;  // 是否在默认CUDA流中执行数据复制
  cuda_options.has_user_compute_stream = 0;
  cuda_options.default_memory_arena_cfg = nullptr;
  /* // 清空 provider 列表（仅某些版本支持）
  #if ORT_API_VERSION >= 14
    session_options
        .DisableCpuMemArena();  // 禁用 CPU 内存池（防止 CPU provider 工作）
  #endif */
  // session_options.DisablePerSessionThreads();  // 可选：关闭多线程 CPU 调度
  session_options.AppendExecutionProvider_CUDA(cuda_options);
  // session_options.SetGraphOptimizationLevel(
  //     GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  Session = std::make_unique<Ort::Session>(env, extractorPath.c_str(),
                                           session_options);

  // Initial Extractor
  size_t numInputNodes = Session->GetInputCount();
  InputNodeNames.reserve(numInputNodes);
  for (size_t i = 0; i < numInputNodes; i++) {
    InputNodeNames.emplace_back(
        strdup(Session->GetInputNameAllocated(i, allocator).get()));
    InputNodeShapes.emplace_back(
        Session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }

  size_t numOutputNodes = Session->GetOutputCount();
  OutputNodeNames.reserve(numOutputNodes);
  for (size_t i = 0; i < numOutputNodes; i++) {
    OutputNodeNames.emplace_back(
        strdup(Session->GetOutputNameAllocated(i, allocator).get()));
    OutputNodeShapes.emplace_back(
        Session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
}

/* void Extractor_DPL::initialize(const std::string &extractorPath,
                               int extractor_type_) {
  extractor_type = extractor_type_;

  // 1️⃣ Env，旧版本 ORT 使用普通构造
  env = Ort::Env(ORT_LOGGING_LEVEL_WARNING,
                 "LightGlueDecoupleOnnxRunner Extractor");

  // 2️⃣ SessionOptions 配置
  session_options = Ort::SessionOptions();
  session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  // 禁用 CPU provider 的内存池和 per-session threads，避免旧版本线程池报错
#if ORT_API_VERSION >= 14
  session_options.DisableCpuMemArena();  // 禁用 CPU 内存池
#endif
  // 这里不要调用 DisablePerSessionThreads()，保持默认即可
  // session_options.EnablePerSessionThreads(); // 可以显式启用 per-session
  // threads（可选）

  // 设置 GPU provider
  OrtCUDAProviderOptions cuda_options{};
  cuda_options.device_id = 0;
  cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
  cuda_options.gpu_mem_limit = 0;
  cuda_options.arena_extend_strategy = 1;      // GPU Arena 扩展策略
  cuda_options.do_copy_in_default_stream = 1;  // 在默认 CUDA stream 执行拷贝
  cuda_options.has_user_compute_stream = 0;
  cuda_options.default_memory_arena_cfg = nullptr;

  session_options.AppendExecutionProvider_CUDA(cuda_options);

  // 3️⃣ 创建 Session
  Session = std::make_unique<Ort::Session>(env, extractorPath.c_str(),
                                           session_options);

  // 4️⃣ 初始化输入节点信息
  size_t numInputNodes = Session->GetInputCount();
  InputNodeNames.reserve(numInputNodes);
  InputNodeShapes.reserve(numInputNodes);
  for (size_t i = 0; i < numInputNodes; i++) {
    InputNodeNames.emplace_back(
        strdup(Session->GetInputNameAllocated(i, allocator).get()));
    InputNodeShapes.emplace_back(
        Session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }

  // 5️⃣ 初始化输出节点信息
  size_t numOutputNodes = Session->GetOutputCount();
  OutputNodeNames.reserve(numOutputNodes);
  OutputNodeShapes.reserve(numOutputNodes);
  for (size_t i = 0; i < numOutputNodes; i++) {
    OutputNodeNames.emplace_back(
        strdup(Session->GetOutputNameAllocated(i, allocator).get()));
    OutputNodeShapes.emplace_back(
        Session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
} */

cv::Mat Extractor_DPL::pre_process(const cv::Mat &Image, float &scale) {
  float temp_scale = scale;
  cv::Mat tempImage = Image.clone();
  std::string fn = "max";
  std::string interp = "area";
  cv::Mat resize_img =
      ResizeImage(tempImage, IMAGE_SIZE_DPL, scale, fn, interp);
  cv::Mat resultImage = NormalizeImage(resize_img);
  if (extractor_type == SUPERPOINT && tempImage.channels() == 3) {
    std::cout << "[INFO] ExtractorType Superpoint turn RGB to Grayscale"
              << std::endl;
    resultImage = RGB2Grayscale(resultImage);
  }
  return resultImage;
}

namespace ort_utils {

template <typename T>
Ort::Value CreateInputTensor(
    const T *data,                      // CPU 内存数据指针
    size_t num_elements,                // 元素数量
    const std::vector<int64_t> &shape,  // 张量 shape
    bool use_cuda = false,              // 是否使用 GPU
    int device_id = 0,                  // GPU 设备 ID
    T *gpu_ptr = nullptr  // 可选：外部提供的 GPU 内存指针
) {
  if (!use_cuda) {
    // CPU tensor
    auto mem_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU);

    return Ort::Value::CreateTensor<T>(mem_info, const_cast<T *>(data),
                                       num_elements, shape.data(),
                                       shape.size());
  } else {
    // GPU tensor
    // 如果外部没有提供 GPU 内存，就自动分配（RAII 交给调用者）
    bool need_free = false;
    if (gpu_ptr == nullptr) {
      cudaSetDevice(device_id);
      size_t bytes = num_elements * sizeof(T);
      cudaMalloc(&gpu_ptr, bytes);
      cudaMemcpy(gpu_ptr, data, bytes, cudaMemcpyHostToDevice);
      need_free = true;  // 标记自己分配的，需要释放
    }

    Ort::MemoryInfo mem_info("Cuda", OrtAllocatorType::OrtDeviceAllocator,
                             device_id, OrtMemType::OrtMemTypeDefault);

    Ort::Value tensor = Ort::Value::CreateTensor<T>(
        mem_info, gpu_ptr, num_elements, shape.data(), shape.size());

    // 如果自己分配了 GPU 内存，可以用智能指针在函数外释放
    if (need_free) {
      // 用 lambda 包装释放函数
      auto deleter = [](T *p) {
        if (p) cudaFree(p);
      };
      // 这里不直接释放，返回后交给调用者管理
      // 调用者可以用 std::unique_ptr<T[], decltype(deleter)> 自动管理
    }

    return tensor;
  }
}

/**
 * @brief 批量创建输入张量（CPU/GPU），支持外部 GPU 内存管理
 * @tparam T 数据类型
 * @param datas CPU 数据指针数组
 * @param num_elements 每个张量元素数量数组
 * @param shapes 每个张量 shape
 * @param use_cuda 是否使用 GPU
 * @param device_id GPU 设备 ID
 * @param gpu_ptrs 可选：外部 GPU 内存指针数组，长度和 datas 相同
 * @return std::vector<Ort::Value> 输入张量
 */
template <typename T>
std::vector<Ort::Value> CreateInputTensors(
    const std::vector<const T *> &datas,
    const std::vector<size_t> &num_elements,
    const std::vector<std::vector<int64_t>> &shapes, bool use_cuda = false,
    int device_id = 0,
    const std::vector<T *> &gpu_ptrs = {})  // 可选外部 GPU 指针
{
  std::vector<Ort::Value> input_tensors;
  size_t n = datas.size();
  input_tensors.reserve(n);

  for (size_t i = 0; i < n; ++i) {
    T *gpu_ptr = nullptr;
    if (use_cuda && !gpu_ptrs.empty()) {
      // 使用外部 GPU 指针
      gpu_ptr = gpu_ptrs[i];
    }

    input_tensors.push_back(CreateInputTensor<T>(
        datas[i], num_elements[i], shapes[i], use_cuda, device_id, gpu_ptr));
  }

  return input_tensors;
}
}  // namespace ort_utils

std::pair<std::vector<cv::Point2f>, float *>
Extractor_DPL::extract_featurepoints(const cv::Mat &image) {
  int InputTensorSize;
  if (extractor_type == SUPERPOINT) {
    InputNodeShapes[0] = {1, 1, image.size().height, image.size().width};
  } else if (extractor_type == DISK) {
    InputNodeShapes[0] = {1, 3, image.size().height, image.size().width};
  }

  InputTensorSize = InputNodeShapes[0][0] * InputNodeShapes[0][1] *
                    InputNodeShapes[0][2] * InputNodeShapes[0][3];

  std::vector<float> srcInputTensorValues(InputTensorSize);

  if (extractor_type == SUPERPOINT) {
    srcInputTensorValues.assign(image.begin<float>(), image.end<float>());
  } else {
    int height = image.rows;
    int width = image.cols;
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        cv::Vec3f pixel = image.at<cv::Vec3f>(y, x);  // RGB
        srcInputTensorValues[y * width + x] = pixel[2];
        srcInputTensorValues[height * width + y * width + x] = pixel[1];
        srcInputTensorValues[2 * height * width + y * width + x] = pixel[0];
      }
    }
  }

  auto memory_info_handler = Ort::MemoryInfo::CreateCpu(
      OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU);
  std::vector<Ort::Value> input_tensors;
  input_tensors.push_back(Ort::Value::CreateTensor<float>(
      memory_info_handler, srcInputTensorValues.data(),
      srcInputTensorValues.size(), InputNodeShapes[0].data(),
      InputNodeShapes[0].size()));
  /* bool use_cuda_only = false;
#ifdef USE_CUDA
  use_cuda_only = true;
#else
  use_cuda_only = false;
#endif

  // 创建 GPU 内存并交给 unique_ptr 管理
  float *gpu_ptr = nullptr;
  size_t num_elements = srcInputTensorValues.size();
  cudaMalloc(&gpu_ptr, num_elements * sizeof(float));
  cudaMemcpy(gpu_ptr, srcInputTensorValues.data(), num_elements * sizeof(float),
             cudaMemcpyHostToDevice);
  auto deleter = [](float *p) {
    if (p) cudaFree(p);
  };
  std::unique_ptr<float, decltype(deleter)> gpu_mem(gpu_ptr, deleter);
  auto input_tensor = ort_utils::CreateInputTensor<float>(
      srcInputTensorValues.data(), num_elements, InputNodeShapes[0],
      use_cuda_only, 0, gpu_ptr);
  std::vector<Ort::Value> input_tensors;
  input_tensors.emplace_back(std::move(input_tensor)); */

  auto output_tensor = Session->Run(
      Ort::RunOptions{nullptr}, InputNodeNames.data(), input_tensors.data(),
      input_tensors.size(), OutputNodeNames.data(), OutputNodeNames.size());

  for (auto &tensor : output_tensor) {
    if (!tensor.IsTensor() || !tensor.HasValue()) {
    }
  }

  outputtensors.emplace_back(std::move(output_tensor));
  std::pair<std::vector<cv::Point2f>, float *> result_pts_descriptors =
      post_process(std::move(outputtensors[0]));

  outputtensors.clear();

  return result_pts_descriptors;
}

std::pair<std::vector<cv::Point2f>, float *> Extractor_DPL::post_process(
    std::vector<Ort::Value> tensor) {
  std::pair<std::vector<cv::Point2f>, float *> extractor_result;
  std::vector<int64_t> kpts_Shape =
      tensor[0].GetTensorTypeAndShapeInfo().GetShape();
  int64_t *kpts = (int64_t *)tensor[0].GetTensorMutableData<void>();

  std::vector<int64_t> score_Shape =
      tensor[1].GetTensorTypeAndShapeInfo().GetShape();
  float *scores = (float *)tensor[1].GetTensorMutableData<void>();

  std::vector<int64_t> descriptors_Shape =
      tensor[2].GetTensorTypeAndShapeInfo().GetShape();
  float *desc = (float *)tensor[2].GetTensorMutableData<void>();
  std::vector<cv::Point2f> kpts_f;
  for (int i = 0; i < kpts_Shape[1] * 2; i += 2) {
    kpts_f.emplace_back(cv::Point2f(kpts[i], kpts[i + 1]));
  }

  extractor_result.first = kpts_f;
  extractor_result.second = desc;
  return extractor_result;
}

void Matcher_DPL::initialize(const std::string &matcherPath,
                             int extractor_type_, float matchThresh_) {
  matchThresh = matchThresh_;
  cout << "match threshold = " << matchThresh << endl;
  extractor_type = extractor_type_;

  env = Ort::Env(ORT_LOGGING_LEVEL_WARNING,
                 "LightGlueDecoupleOnnxRunner Matcher");
  session_options = Ort::SessionOptions();
  session_options.SetInterOpNumThreads(std::thread::hardware_concurrency());
  session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);

  // use gpu
  OrtCUDAProviderOptions cuda_options{};
  cuda_options.device_id = 0;
  cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
  cuda_options.gpu_mem_limit = 0;
  cuda_options.arena_extend_strategy = 1;  // 设置GPU内存管理中的Arena扩展策略
  cuda_options.do_copy_in_default_stream = 1;  // 是否在默认CUDA流中执行数据复制
  cuda_options.has_user_compute_stream = 0;
  cuda_options.default_memory_arena_cfg = nullptr;

  session_options.AppendExecutionProvider_CUDA(cuda_options);
  // session_options.SetGraphOptimizationLevel(
  //     GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  Session =
      std::make_unique<Ort::Session>(env, matcherPath.c_str(), session_options);

  // Initial Extractor
  size_t numInputNodes = Session->GetInputCount();
  InputNodeNames.reserve(numInputNodes);
  for (size_t i = 0; i < numInputNodes; i++) {
    InputNodeNames.emplace_back(
        strdup(Session->GetInputNameAllocated(i, allocator).get()));
    InputNodeShapes.emplace_back(
        Session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }

  size_t numOutputNodes = Session->GetOutputCount();
  OutputNodeNames.reserve(numOutputNodes);
  for (size_t i = 0; i < numOutputNodes; i++) {
    OutputNodeNames.emplace_back(
        strdup(Session->GetOutputNameAllocated(i, allocator).get()));
    OutputNodeShapes.emplace_back(
        Session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
}

/* void Matcher_DPL::initialize(const std::string &matcherPath,
                             int extractor_type_, float matchThresh_) {
  matchThresh = matchThresh_;
  std::cout << "match threshold = " << matchThresh << std::endl;
  extractor_type = extractor_type_;

  // 1️⃣ Env，旧版本 ORT 使用普通构造
  env = Ort::Env(ORT_LOGGING_LEVEL_WARNING,
                 "LightGlueDecoupleOnnxRunner Matcher");

  // 2️⃣ SessionOptions 配置
  session_options = Ort::SessionOptions();
  session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  // 禁用 CPU provider 内存池，避免旧版本线程池报错
#if ORT_API_VERSION >= 14
  session_options.DisableCpuMemArena();
#endif
  // 保持默认 per-session threads，避免线程池报错
  // session_options.EnablePerSessionThreads(); // 可选显式启用

  // 3️⃣ CUDA Provider 配置
  OrtCUDAProviderOptions cuda_options{};
  cuda_options.device_id = 0;
  cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
  cuda_options.gpu_mem_limit = 0;
  cuda_options.arena_extend_strategy = 1;      // GPU Arena 扩展策略
  cuda_options.do_copy_in_default_stream = 1;  // 在默认 CUDA stream 执行拷贝
  cuda_options.has_user_compute_stream = 0;
  cuda_options.default_memory_arena_cfg = nullptr;

  session_options.AppendExecutionProvider_CUDA(cuda_options);

  // 4️⃣ 创建 Session
  Session =
      std::make_unique<Ort::Session>(env, matcherPath.c_str(), session_options);

  // 5️⃣ 初始化输入节点信息
  size_t numInputNodes = Session->GetInputCount();
  InputNodeNames.reserve(numInputNodes);
  InputNodeShapes.reserve(numInputNodes);
  for (size_t i = 0; i < numInputNodes; i++) {
    InputNodeNames.emplace_back(
        strdup(Session->GetInputNameAllocated(i, allocator).get()));
    InputNodeShapes.emplace_back(
        Session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }

  // 6️⃣ 初始化输出节点信息
  size_t numOutputNodes = Session->GetOutputCount();
  OutputNodeNames.reserve(numOutputNodes);
  OutputNodeShapes.reserve(numOutputNodes);
  for (size_t i = 0; i < numOutputNodes; i++) {
    OutputNodeNames.emplace_back(
        strdup(Session->GetOutputNameAllocated(i, allocator).get()));
    OutputNodeShapes.emplace_back(
        Session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
} */

Matcher_DPL::Matcher_DPL(unsigned int _num_threads)
    : num_threads(_num_threads) {}

std::vector<cv::Point2f> Matcher_DPL::pre_process(std::vector<cv::Point2f> kpts,
                                                  int h, int w) {
  return NormalizeKeypoints(kpts, h, w);
}

std::vector<std::pair<int, int>> Matcher_DPL::match_featurepoints(
    std::vector<cv::Point2f> kpts0, std::vector<cv::Point2f> kpts1,
    float *desc0, float *desc1) {
  InputNodeShapes[0] = {1, static_cast<int>(kpts0.size()), 2};
  InputNodeShapes[1] = {1, static_cast<int>(kpts1.size()), 2};
  if (extractor_type == SUPERPOINT) {
    InputNodeShapes[2] = {1, static_cast<int>(kpts0.size()), 256};
    InputNodeShapes[3] = {1, static_cast<int>(kpts1.size()), 256};
  } else {
    InputNodeShapes[2] = {1, static_cast<int>(kpts0.size()), 128};
    InputNodeShapes[3] = {1, static_cast<int>(kpts1.size()), 128};
  }

  auto memory_info_handler = Ort::MemoryInfo::CreateCpu(
      OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU);

  float *kpts0_data = new float[kpts0.size() * 2];
  float *kpts1_data = new float[kpts1.size() * 2];

  for (size_t i = 0; i < kpts0.size(); ++i) {
    kpts0_data[i * 2] = kpts0[i].x;
    kpts0_data[i * 2 + 1] = kpts0[i].y;
  }
  for (size_t i = 0; i < kpts1.size(); ++i) {
    kpts1_data[i * 2] = kpts1[i].x;
    kpts1_data[i * 2 + 1] = kpts1[i].y;
  }

  std::vector<Ort::Value> input_tensors;
  input_tensors.push_back(Ort::Value::CreateTensor<float>(
      memory_info_handler, kpts0_data, kpts0.size() * 2 * sizeof(float),
      InputNodeShapes[0].data(), InputNodeShapes[0].size()));
  input_tensors.push_back(Ort::Value::CreateTensor<float>(
      memory_info_handler, kpts1_data, kpts1.size() * 2 * sizeof(float),
      InputNodeShapes[1].data(), InputNodeShapes[1].size()));
  input_tensors.push_back(Ort::Value::CreateTensor<float>(
      memory_info_handler, desc0, kpts0.size() * 256 * sizeof(float),
      InputNodeShapes[2].data(), InputNodeShapes[2].size()));
  input_tensors.push_back(Ort::Value::CreateTensor<float>(
      memory_info_handler, desc1, kpts1.size() * 256 * sizeof(float),
      InputNodeShapes[3].data(), InputNodeShapes[3].size()));

  /* bool use_cuda_only = false;
#ifdef USE_CUDA
  use_cuda_only = true;
#else
  use_cuda_only = false;
#endif
  std::vector<const float *> datas = {kpts0_data, kpts1_data, desc0, desc1};
  std::vector<size_t> num_elements = {kpts0.size() * 2, kpts1.size() * 2,
                                      kpts0.size() * 256, kpts1.size() * 256};
  std::vector<std::vector<int64_t>> shapes = {
      InputNodeShapes[0], InputNodeShapes[1], InputNodeShapes[2],
      InputNodeShapes[3]};
  // 先分配 GPU 显存并用 unique_ptr 管理
  std::vector<float *> gpu_ptrs(datas.size(), nullptr);
  std::vector<std::unique_ptr<float, decltype(&cudaFree)>> gpu_raii;

  for (size_t i = 0; i < datas.size(); ++i) {
    float *ptr = nullptr;
    cudaMalloc(&ptr, num_elements[i] * sizeof(float));
    cudaMemcpy(ptr, datas[i], num_elements[i] * sizeof(float),
               cudaMemcpyHostToDevice);
    gpu_ptrs[i] = ptr;
    gpu_raii.emplace_back(ptr, &cudaFree);  // RAII 自动释放
  }

  // 批量创建 Ort tensor
  auto input_tensors = ort_utils::CreateInputTensors<float>(
      datas, num_elements, shapes, use_cuda_only, 0, gpu_ptrs); */

  auto output_tensor = Session->Run(
      Ort::RunOptions{nullptr}, InputNodeNames.data(), input_tensors.data(),
      input_tensors.size(), OutputNodeNames.data(), OutputNodeNames.size());

  for (auto &tensor : output_tensor) {
    if (!tensor.IsTensor() || !tensor.HasValue()) {
      std::cerr << "[ERROR] Inference output tensor is not a tensor or don't "
                   "have value"
                << std::endl;
    }
  }
  outputtensors = std::move(output_tensor);

  std::vector<std::pair<int, int>> result_matches = post_process();

  outputtensors.clear();

  return result_matches;
}

std::vector<std::pair<int, int>> Matcher_DPL::post_process() {
  std::vector<std::pair<int, int>> good_matches;
  // load date from tensor
  std::vector<int64_t> matches_Shape =
      outputtensors[0].GetTensorTypeAndShapeInfo().GetShape();
  int64_t *matches = (int64_t *)outputtensors[0].GetTensorMutableData<void>();
  std::vector<int64_t> mscore_Shape =
      outputtensors[1].GetTensorTypeAndShapeInfo().GetShape();
  float *mscores = (float *)outputtensors[1].GetTensorMutableData<void>();
  for (int i = 0; i < matches_Shape[0]; i++) {
    if (mscores[i] > this->matchThresh) {
      good_matches.emplace_back(
          std::make_pair(matches[i * 2], matches[i * 2 + 1]));
    }
  }
  return good_matches;
}