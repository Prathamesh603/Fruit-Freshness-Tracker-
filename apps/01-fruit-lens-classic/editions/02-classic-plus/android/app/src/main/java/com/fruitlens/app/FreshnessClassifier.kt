package com.fruitlens.app

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest

data class ScanResult(val label: FruitLabel, val score: Float, val accepted: Boolean, val processingMs: Long, val confident: Boolean = false, val freshScore: Float? = null)

/** Owned by one worker thread. Models and tensor buffers are reused between scans. */
class FreshnessClassifier(context: Context) : AutoCloseable {
    val spec = ModelSpec.parse(context.assets.open("model/model.json").bufferedReader().use { it.readText() })
    private val interpreter: Interpreter
    private val input: ByteBuffer
    private val output: ByteBuffer
    private val inputTensor: Tensor
    private val outputTensor: Tensor

    init {
        val model = context.assets.open("model/${spec.modelFile}").use { it.readBytes() }
        val hash = MessageDigest.getInstance("SHA-256").digest(model).joinToString("") { "%02x".format(it.toInt() and 255) }
        require(hash == spec.sha256) { "Model checksum differs from the manifest." }
        val buffer = ByteBuffer.allocateDirect(model.size).order(ByteOrder.nativeOrder()).apply { put(model); rewind() }
        val loaded = Interpreter(buffer, Interpreter.Options().setNumThreads(2).setUseXNNPACK(true))
        try {
            require(loaded.inputTensorCount == 1 && loaded.outputTensorCount == 1)
            inputTensor = loaded.getInputTensor(0)
            outputTensor = loaded.getOutputTensor(0)
            require(inputTensor.shape().contentEquals(intArrayOf(1, spec.height, spec.width, 3))) {
                "Input shape differs from model.json."
            }
            require(outputTensor.shape().contentEquals(intArrayOf(1, spec.labels.size))) {
                "Output shape differs from model labels."
            }
            listOf(inputTensor, outputTensor).forEach {
                require(it.dataType() in setOf(DataType.FLOAT32, DataType.INT8, DataType.UINT8)) { "Unsupported tensor dtype." }
                if (it.dataType() != DataType.FLOAT32) require(it.quantizationParams().scale > 0f)
            }
            input = ByteBuffer.allocateDirect(inputTensor.numBytes()).order(ByteOrder.nativeOrder())
            output = ByteBuffer.allocateDirect(outputTensor.numBytes()).order(ByteOrder.nativeOrder())
            interpreter = loaded
        } catch (error: Exception) {
            loaded.close()
            throw error
        }
    }

    fun classify(bitmap: Bitmap): ScanResult {
        val start = System.nanoTime()
        val resized = Bitmap.createScaledBitmap(bitmap, spec.width, spec.height, spec.interpolation == "bilinear")
        try {
            val pixels = IntArray(spec.width * spec.height)
            resized.getPixels(pixels, 0, spec.width, 0, 0, spec.width, spec.height)
            input.rewind()
            val quant = inputTensor.quantizationParams()
            pixels.forEach { pixel ->
                for (channel in 0..2) {
                    val value = (pixel shr (16 - 8 * channel)) and 255
                    val normalized = ScoreMath.normalize(value, spec.mean[channel], spec.std[channel])
                    if (inputTensor.dataType() == DataType.FLOAT32) input.putFloat(normalized)
                    else input.put(ScoreMath.quantize(normalized, quant.scale, quant.zeroPoint,
                        inputTensor.dataType() == DataType.INT8).toByte())
                }
            }
            input.rewind()
            output.rewind()
            interpreter.run(input, output)
            output.rewind()
            val outputQuant = outputTensor.quantizationParams()
            val values = FloatArray(spec.labels.size) {
                if (outputTensor.dataType() == DataType.FLOAT32) output.float
                else {
                    val raw = output.get().toInt()
                    val number = if (outputTensor.dataType() == DataType.UINT8) raw and 255 else raw
                    (number - outputQuant.zeroPoint) * outputQuant.scale
                }
            }
            val scores = ScoreMath.probabilities(values, spec.scoreType == "logits")
            val best = scores.indices.maxBy { scores[it] }
            val fruit = spec.labels[best].fruit
            val fresh = spec.labels.indexOfFirst { it.fruit == fruit && it.condition == "fresh" }
            val rotten = spec.labels.indexOfFirst { it.fruit == fruit && it.condition == "rotten" }
            val freshScore = if (fresh >= 0 && rotten >= 0) ScoreMath.freshScore(scores[fresh], scores[rotten]) else null
            return ScanResult(spec.labels[best], scores[best],
                spec.thresholdValidated && scores[best] >= spec.minScore,
                (System.nanoTime() - start) / 1_000_000, scores[best] >= spec.minScore, freshScore)
        } finally {
            if (resized !== bitmap) resized.recycle()
        }
    }

    override fun close() = interpreter.close()
}
