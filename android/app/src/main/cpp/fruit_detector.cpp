#include <jni.h>
#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <net.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
constexpr int kInput = 416;
constexpr int kClasses = 80;
constexpr int kBins = 8;
constexpr float kThreshold = 0.4f;
struct Box { int label; float score, left, top, right, bottom; };

bool supported(int label) { return label == 46 || label == 47 || label == 49; }

float overlap(const Box& a, const Box& b) {
    const float intersection = std::max(0.f, std::min(a.right, b.right) - std::max(a.left, b.left)) *
                               std::max(0.f, std::min(a.bottom, b.bottom) - std::max(a.top, b.top));
    const float sum = (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top);
    return intersection / std::max(1e-6f, sum - intersection);
}

float distance(const float* logits) {
    const float maximum = *std::max_element(logits, logits + kBins);
    float weighted = 0.f, total = 0.f;
    for (int bin = 0; bin < kBins; ++bin) {
        const float weight = std::exp(logits[bin] - maximum);
        total += weight;
        weighted += bin * weight;
    }
    return weighted / total;
}

void fail(JNIEnv* env, const char* message) { env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), message); }

class Detector {
public:
    ncnn::Net net;
    explicit Detector(AAssetManager* assets) {
        net.opt.num_threads = 2;
        net.opt.use_vulkan_compute = false;
        net.opt.use_fp16_storage = true;
        if (net.load_param(assets, "detector/nanodet.param") || net.load_model(assets, "detector/nanodet.bin"))
            throw std::runtime_error("Could not load NanoDet-Plus detection weights.");
    }

    std::vector<Box> run(const unsigned char* pixels, int width, int height, int stride) {
        const float scale = static_cast<float>(kInput) / std::max(width, height);
        const int resizedWidth = std::max(1, static_cast<int>(std::round(width * scale)));
        const int resizedHeight = std::max(1, static_cast<int>(std::round(height * scale)));
        const int leftPad = (kInput - resizedWidth) / 2;
        const int topPad = (kInput - resizedHeight) / 2;
        ncnn::Mat resized = ncnn::Mat::from_pixels_resize(pixels, ncnn::Mat::PIXEL_RGBA2BGR,
            width, height, stride, resizedWidth, resizedHeight);
        ncnn::Mat input;
        ncnn::copy_make_border(resized, input, topPad, kInput - resizedHeight - topPad,
            leftPad, kInput - resizedWidth - leftPad, ncnn::BORDER_CONSTANT, 0.f);
        const float mean[3] = {103.53f, 116.28f, 123.675f};
        const float norm[3] = {1.f / 57.375f, 1.f / 57.12f, 1.f / 58.395f};
        input.substract_mean_normalize(mean, norm);
        auto extractor = net.create_extractor();
        ncnn::Mat output;
        if (extractor.input("data", input) || extractor.extract("output", output))
            throw std::runtime_error("Fruit detection inference failed.");
        if (output.w != kClasses + 4 * kBins || output.h != 3598)
            throw std::runtime_error("Unexpected NanoDet-Plus output dimensions.");
        std::vector<Box> candidates;
        int row = 0;
        for (int step : {8, 16, 32, 64}) {
            const int grid = (kInput + step - 1) / step;
            for (int y = 0; y < grid; ++y) {
                for (int x = 0; x < grid; ++x, ++row) {
                    const float* prediction = output.row(row);
                    const int label = static_cast<int>(std::max_element(prediction, prediction + kClasses) - prediction);
                    if (!supported(label) || !std::isfinite(prediction[label]) || prediction[label] < kThreshold) continue;
                    const float* offsets = prediction + kClasses;
                    const float x1 = (x * step - distance(offsets) * step - leftPad) / resizedWidth;
                    const float y1 = (y * step - distance(offsets + kBins) * step - topPad) / resizedHeight;
                    const float x2 = (x * step + distance(offsets + 2 * kBins) * step - leftPad) / resizedWidth;
                    const float y2 = (y * step + distance(offsets + 3 * kBins) * step - topPad) / resizedHeight;
                    if (!std::isfinite(x1 + y1 + x2 + y2)) continue;
                    Box box{label, prediction[label], std::clamp(x1, 0.f, 1.f), std::clamp(y1, 0.f, 1.f),
                            std::clamp(x2, 0.f, 1.f), std::clamp(y2, 0.f, 1.f)};
                    if (box.right - box.left > 0.02f && box.bottom - box.top > 0.02f) candidates.push_back(box);
                }
            }
        }
        std::sort(candidates.begin(), candidates.end(), [](const Box& a, const Box& b) { return a.score > b.score; });
        std::vector<Box> picked;
        for (const auto& candidate : candidates) {
            bool duplicate = false;
            for (const auto& prior : picked) {
                if (prior.label == candidate.label && overlap(prior, candidate) > 0.5f) { duplicate = true; break; }
            }
            if (!duplicate) picked.push_back(candidate);
            if (picked.size() == 5) break;
        }
        return picked;
    }
};
} // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_fruitlens_app_FruitDetector_nativeCreate(JNIEnv* env, jobject, jobject assetManager) {
    try { return reinterpret_cast<jlong>(new Detector(AAssetManager_fromJava(env, assetManager))); }
    catch (const std::exception& error) { fail(env, error.what()); return 0; }
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_fruitlens_app_FruitDetector_nativeDetect(JNIEnv* env, jobject, jlong handle, jobject bitmap) {
    AndroidBitmapInfo info{};
    if (!handle || AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS ||
        info.format != ANDROID_BITMAP_FORMAT_RGBA_8888 || info.width == 0 || info.height == 0) {
        fail(env, "Unsupported detection image."); return nullptr;
    }
    void* pixels = nullptr;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS) {
        fail(env, "Could not access camera image pixels."); return nullptr;
    }
    std::vector<Box> boxes;
    try { boxes = reinterpret_cast<Detector*>(handle)->run(static_cast<unsigned char*>(pixels), info.width, info.height, info.stride); }
    catch (const std::exception& error) {
        AndroidBitmap_unlockPixels(env, bitmap); fail(env, error.what()); return nullptr;
    }
    AndroidBitmap_unlockPixels(env, bitmap);
    std::vector<float> values;
    for (const auto& box : boxes) values.insert(values.end(), {static_cast<float>(box.label), box.score, box.left, box.top, box.right, box.bottom});
    jfloatArray result = env->NewFloatArray(static_cast<jsize>(values.size()));
    if (result && !values.empty()) env->SetFloatArrayRegion(result, 0, static_cast<jsize>(values.size()), values.data());
    return result;
}

extern "C" JNIEXPORT void JNICALL
Java_com_fruitlens_app_FruitDetector_nativeDestroy(JNIEnv*, jobject, jlong handle) { delete reinterpret_cast<Detector*>(handle); }
