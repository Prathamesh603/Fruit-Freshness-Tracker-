package com.fruitlens.app

import org.junit.Assert.*
import org.junit.Test

class FreshnessDisplayTest {
    @Test fun usesSameFruitPairRatherThanWinningClassConfidence() {
        assertEquals(0.9f, ScoreMath.freshScore(0.09f, 0.01f)!!, 0.00001f)
        assertEquals(0.1f, ScoreMath.freshScore(0.01f, 0.09f)!!, 0.00001f)
        assertNull(ScoreMath.freshScore(0f, 0f))
        assertNull(ScoreMath.freshScore(Float.NaN, 1f))
    }
    @Test fun paletteRunsFromRedThroughAmberToGreen() {
        assertEquals(0xFFC62828.toInt(), FreshnessDisplay.tint(0f, true, true))
        assertEquals(0xFFF9A825.toInt(), FreshnessDisplay.tint(0.5f, true, true))
        assertEquals(0xFF2E7D32.toInt(), FreshnessDisplay.tint(1f, true, true))
        assertNotEquals(FreshnessDisplay.tint(0.9f, true, true), FreshnessDisplay.tint(0.1f, true, true))
    }
    @Test fun uncertainOrInvalidAssessmentsStayNeutral() {
        assertEquals(FreshnessDisplay.NEUTRAL, FreshnessDisplay.tint(1f, false, true))
        assertEquals(FreshnessDisplay.NEUTRAL, FreshnessDisplay.tint(1f, true, false))
        assertEquals(FreshnessDisplay.NEUTRAL, FreshnessDisplay.tint(null, true, true))
        assertEquals(FreshnessDisplay.NEUTRAL, FreshnessDisplay.tint(Float.NaN, true, true))
    }
}
