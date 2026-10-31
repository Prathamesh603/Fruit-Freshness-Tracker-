package com.fruitlens.app

import org.junit.Assert.*
import org.junit.Test

class ScoreMathTest {
    @Test fun normalizationUsesRawRgbContract() {
        assertEquals(-1f, ScoreMath.normalize(0, 127.5f, 127.5f), 0.00001f)
        assertEquals(1f, ScoreMath.normalize(255, 127.5f, 127.5f), 0.00001f)
        assertEquals(1f, ScoreMath.normalize(255, 0f, 255f), 0.00001f)
    }
    @Test fun quantizationRoundsAndClamps() {
        assertEquals(255, ScoreMath.quantize(10f, 0.01f, 0, false))
        assertEquals(-128, ScoreMath.quantize(-10f, 0.01f, 0, true))
        assertEquals(128, ScoreMath.quantize(0f, 0.1f, 128, false))
    }
    @Test fun softmaxHandlesLargeLogitsWithoutOverflow() {
        val scores = ScoreMath.probabilities(floatArrayOf(1000f, 1000f), true)
        assertArrayEquals(floatArrayOf(0.5f, 0.5f), scores, 0.00001f)
    }
    @Test(expected = IllegalArgumentException::class)
    fun rejectsNonFiniteOutput() { ScoreMath.probabilities(floatArrayOf(Float.NaN, 1f), true) }
    @Test(expected = IllegalArgumentException::class)
    fun rejectsLogitsMisdeclaredAsProbabilities() { ScoreMath.probabilities(floatArrayOf(4f, -2f), false) }
}
