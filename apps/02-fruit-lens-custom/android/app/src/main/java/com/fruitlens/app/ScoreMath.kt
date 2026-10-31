package com.fruitlens.app

import kotlin.math.exp
import kotlin.math.roundToInt

/** Shared numerical contract; no Android APIs so these checks can run on the JVM. */
object ScoreMath {
    /** Relative Fresh vs Not fresh score for one fruit, not a freshness percentage. */
    fun freshScore(fresh: Float, notFresh: Float): Float? {
        if (!fresh.isFinite() || !notFresh.isFinite() || fresh < 0f || notFresh < 0f) return null
        val total = fresh + notFresh
        return if (total.isFinite() && total > 0f) (fresh / total).coerceIn(0f, 1f) else null
    }
    fun normalize(pixel: Int, mean: Float, std: Float): Float = (pixel - mean) / std

    fun quantize(real: Float, scale: Float, zeroPoint: Int, signed: Boolean): Int {
        require(scale > 0f && scale.isFinite())
        return (real / scale + zeroPoint).roundToInt().coerceIn(if (signed) -128 else 0, if (signed) 127 else 255)
    }

    fun probabilities(values: FloatArray, logits: Boolean): FloatArray {
        require(values.isNotEmpty() && values.all { it.isFinite() }) { "Model produced invalid scores." }
        if (!logits) {
            require(values.all { it in -0.01f..1.01f } && kotlin.math.abs(values.sum() - 1f) < 0.1f) {
                "Model output does not match the declared probability format."
            }
            return values.map { it.coerceIn(0f, 1f) }.toFloatArray()
        }
        val maximum = values.max()
        val exponents = values.map { exp((it - maximum).toDouble()) }
        val total = exponents.sum()
        return FloatArray(values.size) { (exponents[it] / total).toFloat() }
    }
}
