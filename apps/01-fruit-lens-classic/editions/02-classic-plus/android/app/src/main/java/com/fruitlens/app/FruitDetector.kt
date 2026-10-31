package com.fruitlens.app

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.RectF

data class FruitDetection(val fruit: String, val score: Float, val bounds: RectF)

/** One worker owns native detector creation, inference and destruction. */
class FruitDetector(context: Context) : AutoCloseable {
    private var handle = nativeCreate(context.assets)

    fun detect(bitmap: Bitmap): List<FruitDetection> {
        check(handle != 0L)
        val raw = nativeDetect(handle, bitmap)
        check(raw.size % 6 == 0)
        return raw.asList().chunked(6).map {
            FruitDetection(when (it[0].toInt()) { 46 -> "banana"; 47 -> "apple"; 49 -> "orange"; else -> error("Unsupported fruit label") },
                it[1], RectF(it[2], it[3], it[4], it[5]))
        }
    }

    override fun close() {
        if (handle != 0L) nativeDestroy(handle)
        handle = 0L
    }

    private external fun nativeCreate(assets: AssetManager): Long
    private external fun nativeDetect(handle: Long, bitmap: Bitmap): FloatArray
    private external fun nativeDestroy(handle: Long)

    companion object { val names = listOf("apple", "banana", "orange"); init { System.loadLibrary("fruitvision") } }
}
