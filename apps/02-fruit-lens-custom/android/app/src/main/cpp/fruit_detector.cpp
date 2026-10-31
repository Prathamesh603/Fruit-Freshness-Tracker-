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
constexpr int kClasses = 9;
constexpr int kCandidates = 3549;
constexpr float kThreshold = 0.4f;
struct Box { int label; float score, left, top, right, bottom; };

float overlap(const Box& a, const Box& b) {
    const float intersection = std::max(0.f, std::min(a.right, b.right) - std::max(a.left, b.left)) *
                               std::max(0.f, std::min(a.bottom, b.bottom) - std::max(a.top, b.top));
    const float sum = (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top);
    return intersection / std::max(1e-6f, sum - intersection);
}

void fail(JNIEnv* env, const char* message) { env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), message); }

class Detector {
public:
    ncnn::Net net;
    explicit Detector(AAssetManager* assets) {
        net.opt.num_threads = 2;
        net.opt.use_vulkan_compute = false;
        net.opt.use_fp16_storage = true;
        net.opt.use_fp16_arithmetic = false;
        if (net.load_param(assets, "detector/detector.param") || net.load_model(assets, "detector/detector.bin"))
            throw std::runtime_error("Could not load YOLOv8n custom detection weights.");
    }

    std::vector<Box> run(const unsigned char* pixels, int width, int height, int stride) {
        const float scale = static_cast<float>(kInput) / std::max(width, height);
        const int resizedWidth = std::max(1, static_cast<int>(std::round(width * scale)));
        const int resizedHeight = std::max(1, static_cast<int>(std::round(height * scale)));
        const int leftPad = (kInput - resizedWidth) / 2;
        const int topPad = (kInput - resizedHeight) / 2;
        ncnn::Mat resized = ncnn::Mat::from_pixels_resize(pixels, ncnn::Mat::PIXEL_RGBA2RGB,
            width, height, stride, resizedWidth, resizedHeight);
        ncnn::Mat input;
        ncnn::copy_make_border(resized, input, topPad, kInput - resizedHeight - topPad,
            leftPad, kInput - resizedWidth - leftPad, ncnn::BORDER_CONSTANT, 114.f);
        const float norm[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
        input.substract_mean_normalize(nullptr, norm);
        auto extractor = net.create_extractor();
        ncnn::Mat output;
        if (extractor.input("in0", input) || extractor.extract("out0", output))
            throw std::runtime_error("Fruit detection inference failed.");
        if (output.dims != 2 || output.w != kCandidates || output.h != 4 + kClasses)
            throw std::runtime_error("Unexpected YOLOv8n output dimensions.");
        std::vector<Box> candidates;
        for (int i = 0; i < output.w; ++i) {
            int label = 0;
            float score = output.row(4)[i];
            for (int c = 1; c < kClasses; ++c) {
                if (output.row(4 + c)[i] > score) { label = c; score = output.row(4 + c)[i]; }
            }
            if (!std::isfinite(score) || score < kThreshold) continue;
            const float cx = output.row(0)[i], cy = output.row(1)[i];
            const float bw = output.row(2)[i], bh = output.row(3)[i];
            if (!std::isfinite(cx + cy + bw + bh) || bw <= 0.f || bh <= 0.f) continue;
            Box box{label, score,
                std::clamp((cx - bw / 2 - leftPad) / resizedWidth, 0.f, 1.f),
                std::clamp((cy - bh / 2 - topPad) / resizedHeight, 0.f, 1.f),
                std::clamp((cx + bw / 2 - leftPad) / resizedWidth, 0.f, 1.f),
                std::clamp((cy + bh / 2 - topPad) / resizedHeight, 0.f, 1.f)};
            if (box.right - box.left > 0.02f && box.bottom - box.top > 0.02f) candidates.push_back(box);
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
