package com.fruitlens.app

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.Rect
import android.net.Uri
import androidx.exifinterface.media.ExifInterface
import androidx.lifecycle.AndroidViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.io.FileNotFoundException
import java.util.concurrent.Executors
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.atomic.AtomicBoolean

data class ScannerState(
    val loading: Boolean = true,
    val ready: Boolean = false,
    val busy: Boolean = false,
    val modelId: String = "",
    val message: String? = null,
    val results: List<DetectedResult> = emptyList(),
    val photo: Bitmap? = null,
    val processingMs: Long = 0,
)

data class DetectedResult(val detection: FruitDetection, val classification: ScanResult, val id: Int = 0)

class ScannerViewModel(application: Application) : AndroidViewModel(application) {
    private val worker = Executors.newSingleThreadExecutor()
    private val closed = AtomicBoolean(false)
    private val mutableState = MutableStateFlow(ScannerState())
    val state = mutableState.asStateFlow()
    private var classifier: FreshnessClassifier? = null
    private var detector: FruitDetector? = null

    init {
        worker.execute {
            try {
                classifier = FreshnessClassifier(application)
                detector = FruitDetector(application)
                // Pay lazy kernel/allocation setup once, before the scan button becomes available.
                val warmup = Bitmap.createBitmap(416, 416, Bitmap.Config.ARGB_8888)
                try {
                    warmup.eraseColor(android.graphics.Color.GRAY)
                    detector!!.detect(warmup)
                    classifier!!.classify(warmup)
                } finally { warmup.recycle() }
                mutableState.value = ScannerState(loading = false, ready = true, modelId = classifier!!.spec.modelId)
            } catch (_: FileNotFoundException) {
                mutableState.value = ScannerState(loading = false,
                    message = "The model is not installed yet. You can try the camera while training finishes.")
            } catch (error: Exception) {
                mutableState.value = ScannerState(loading = false,
                    message = "Model could not load: ${error.message ?: "unsupported model"}")
            } catch (error: LinkageError) {
                mutableState.value = ScannerState(loading = false,
                    message = "The native detector is unavailable on this device: ${error.message}")
            }
        }
    }

    fun beginCapture(): Boolean {
        if (closed.get() || !mutableState.value.ready || mutableState.value.busy) return false
        mutableState.value = mutableState.value.copy(busy = true, results = emptyList(), photo = null, message = null)
        return true
    }

    fun reportError(message: String) {
        mutableState.value = mutableState.value.copy(busy = false, message = message, results = emptyList())
    }

    fun resetResult() {
        if (!mutableState.value.busy) mutableState.value = mutableState.value.copy(photo = null, results = emptyList(), message = null)
    }

    fun classify(uri: Uri, deleteAfter: Boolean = false) {
        fun cleanCapture() {
            if (deleteAfter && uri.scheme == "file") uri.path?.let { java.io.File(it).delete() }
        }
        if (closed.get()) { cleanCapture(); return }
        if (!mutableState.value.busy && !beginCapture()) return
        try { worker.execute {
            var bitmap: Bitmap? = null
            try {
                bitmap = decode(uri)
                val photo = bitmap
                val start = System.nanoTime()
                val detected = requireNotNull(detector).detect(photo)
                val results = detected.mapNotNull { detection ->
                    val box = detection.bounds
                    val padX = box.width() * 0.06f
                    val padY = box.height() * 0.06f
                    val rect = Rect(((box.left - padX) * photo.width).toInt().coerceIn(0, photo.width - 1),
                        ((box.top - padY) * photo.height).toInt().coerceIn(0, photo.height - 1),
                        kotlin.math.ceil(((box.right + padX) * photo.width).toDouble()).toInt().coerceIn(1, photo.width),
                        kotlin.math.ceil(((box.bottom + padY) * photo.height).toDouble()).toInt().coerceIn(1, photo.height))
                    if (rect.width() < 24 || rect.height() < 24) return@mapNotNull null
                    val crop = Bitmap.createBitmap(photo, rect.left, rect.top, rect.width(), rect.height())
                    try {
                        val classification = requireNotNull(classifier).classify(crop)
                        DetectedResult(detection, classification.copy(accepted = classification.accepted && classification.label.fruit == detection.fruit))
                    } finally { if (crop !== photo) crop.recycle() }
                }
                val numbered = results.mapIndexed { index, result -> result.copy(id = index + 1) }
                mutableState.value = mutableState.value.copy(busy = false,
                    message = if (results.isEmpty()) "No supported fruit was found clearly enough. Move closer in even light. Supported: apple, banana, grape, lemon, mango, orange, pomegranate, strawberry and tomato." else null,
                    results = numbered, photo = photo, processingMs = (System.nanoTime() - start) / 1_000_000)
                bitmap = null // State owns this image until replaced; allow GC instead of recycling a displayed bitmap.
            } catch (error: Exception) {
                reportError("Couldn't assess this photo. ${error.message ?: "Try a clearer image."}")
            } finally {
                bitmap?.recycle()
                cleanCapture()
            }
        } } catch (_: RejectedExecutionException) { cleanCapture() }
    }

    private fun decode(uri: Uri): Bitmap {
        val resolver = getApplication<Application>().contentResolver
        fun stream() = requireNotNull(resolver.openInputStream(uri)) { "Photo could not be opened." }
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        stream().use { BitmapFactory.decodeStream(it, null, bounds) }
        require(bounds.outWidth > 0 && bounds.outHeight > 0) { "Unsupported image." }
        var sample = 1
        while (maxOf(bounds.outWidth, bounds.outHeight) / sample > 1600) sample *= 2
        val options = BitmapFactory.Options().apply { inSampleSize = sample; inPreferredConfig = Bitmap.Config.ARGB_8888 }
        val original = stream().use { BitmapFactory.decodeStream(it, null, options) }
            ?: throw IllegalArgumentException("Photo could not be decoded.")
        val exif = try { stream().use { ExifInterface(it) } } catch (_: Exception) { null }
        val rotation = exif?.rotationDegrees ?: 0
        val flipped = exif?.isFlipped ?: false
        if (rotation == 0 && !flipped) return original
        val matrix = Matrix().apply {
            if (flipped) postScale(-1f, 1f)
            postRotate(rotation.toFloat())
        }
        return Bitmap.createBitmap(original, 0, 0, original.width, original.height, matrix, true)
            .also { if (it !== original) original.recycle() }
    }

    override fun onCleared() {
        closed.set(true)
        // Queue cleanup after any active inference; never close the interpreter while it runs.
        worker.execute { classifier?.close(); detector?.close() }
        worker.shutdown()
    }
}
