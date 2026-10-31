package com.fruitlens.app

import kotlin.math.roundToInt
import java.util.Locale

/** Pure display rules shared by the image overlay and its matching detail card. */
object FreshnessDisplay {
    fun scoreLabel(score: Float): String = when {
        !score.isFinite() || score !in 0f..1f -> "Unavailable"
        score > 0.999f -> ">99.9%"
        else -> String.format(Locale.ROOT, "%.1f%%", score * 100f)
    }
    const val NEUTRAL: Int = -10193781 // #64748B
    private val red = intArrayOf(198, 40, 40)
    private val amber = intArrayOf(249, 168, 37)
    private val green = intArrayOf(46, 125, 50)

    fun conditionName(condition: String): String = when (condition) {
        "fresh" -> "Fresh"
        "rotten", "spoiled" -> "Not fresh"
        "unripe" -> "Unripe"
        else -> "Uncertain"
    }

    fun tint(freshScore: Float?, confident: Boolean, identityAgrees: Boolean, condition: String? = null): Int {
        if (!confident || !identityAgrees) return NEUTRAL
        // Unripe is a separate source class, not a synonym for rotten.
        if (condition == "unripe") return 0xFFF9A825.toInt()
        if (freshScore == null || !freshScore.isFinite() || freshScore !in 0f..1f)
            return NEUTRAL
        val start = if (freshScore < 0.5f) red else amber
        val end = if (freshScore < 0.5f) amber else green
        val fraction = if (freshScore < 0.5f) freshScore * 2 else (freshScore - 0.5f) * 2
        val channels = IntArray(3) { (start[it] + (end[it] - start[it]) * fraction).roundToInt() }
        return (0xFF shl 24) or (channels[0] shl 16) or (channels[1] shl 8) or channels[2]
    }

    fun tint(result: DetectedResult) = tint(result.classification.freshScore,
        result.classification.confident, result.classification.label.fruit == result.detection.fruit,
        result.classification.label.condition)

    fun assessment(result: DetectedResult): String = when {
        !result.classification.confident || result.classification.label.fruit != result.detection.fruit -> "Uncertain"
        else -> conditionName(result.classification.label.condition)
    }
}
