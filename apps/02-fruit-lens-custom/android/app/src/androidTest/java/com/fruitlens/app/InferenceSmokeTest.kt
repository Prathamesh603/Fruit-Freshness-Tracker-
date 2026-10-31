package com.fruitlens.app

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.util.Log
import android.net.Uri
import android.os.ParcelFileDescriptor
import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.junit4.createEmptyComposeRule
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.captureToImage
import androidx.compose.ui.test.onRoot
import androidx.compose.ui.graphics.asAndroidBitmap
import androidx.lifecycle.ViewModelProvider
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Test
import org.junit.Rule
import org.junit.runner.RunWith
import java.io.File

@RunWith(AndroidJUnit4::class)
class InferenceSmokeTest {
    @get:Rule val compose = createEmptyComposeRule()
    private val instrumentation = InstrumentationRegistry.getInstrumentation()

    @Test fun realModelsLoadAndClassifyRepeatedly() {
        val photo = instrumentation.context.assets.open("banana.jpg").use {
            BitmapFactory.decodeStream(it, null, BitmapFactory.Options().apply { inSampleSize = 2; inPreferredConfig = Bitmap.Config.ARGB_8888 })
        }!!
        try {
            FruitDetector(instrumentation.targetContext).use { detector ->
                val start = System.nanoTime()
                val boxes = detector.detect(photo)
                Log.i("FruitLensTest", "detected=${boxes.map { it.fruit to it.score }} milliseconds=${(System.nanoTime() - start) / 1_000_000}")
                assertTrue("Detector should identify the banana fixture", boxes.any { it.fruit == "banana" })
                assertTrue(boxes.all { it.bounds.left >= 0 && it.bounds.right <= 1 && it.bounds.top >= 0 && it.bounds.bottom <= 1 })
                FreshnessClassifier(instrumentation.targetContext).use { classifier ->
                    repeat(3) {
                        val result = classifier.classify(photo)
                        assertTrue(result.score.isFinite() && result.score in 0f..1f)
                        assertTrue(classifier.spec.labels.any { label -> label.id == result.label.id })
                        Log.i("FruitLensTest", "suggestion=${result.label.id} score=${result.score} processingMs=${result.processingMs}")
                        if (!classifier.spec.thresholdValidated) assertFalse(result.accepted)
                    }
                }
            }
        } finally { photo.recycle() }
    }

    @Test fun allEighteenClassifierOutputsMatchExport() {
        val cases = org.json.JSONArray(instrumentation.context.assets.open("classifier/fixtures.json").bufferedReader().use { it.readText() })
        FreshnessClassifier(instrumentation.targetContext).use { classifier ->
            assertEquals(18, classifier.spec.labels.size)
            assertEquals(FruitDetector.names.toSet(), classifier.spec.labels.map { it.fruit }.toSet())
            for (i in 0 until cases.length()) {
                val fixture = cases.getJSONObject(i)
                val image = instrumentation.context.assets.open("classifier/${fixture.getString("file")}").use { BitmapFactory.decodeStream(it) }!!
                try {
                    val result = classifier.classify(image)
                    assertEquals("Output class for ${fixture.getString("file")}", classifier.spec.labels[fixture.getInt("expected_class")].id, result.label.id)
                    assertEquals(fixture.getDouble("expected_score"), result.score.toDouble(), 0.025)
                    assertFalse(result.accepted)
                } finally { image.recycle() }
            }
        }
    }

    @Test fun emptySceneDoesNotBecomeAFruitResult() {
        val bitmap = Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888).apply { eraseColor(Color.WHITE) }
        try {
            FruitDetector(instrumentation.targetContext).use { detector -> assertTrue(detector.detect(bitmap).isEmpty()) }
        } finally { bitmap.recycle() }
    }

    @Test fun capturedPhotoPipelineProducesAnExperimentalResult() {
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            lateinit var model: ScannerViewModel
            scenario.onActivity { activity -> model = ViewModelProvider(activity)[ScannerViewModel::class.java] }
            val loadDeadline = System.nanoTime() + 30_000_000_000
            while (model.state.value.loading && System.nanoTime() < loadDeadline) Thread.sleep(100)
            assertTrue(model.state.value.message, model.state.value.ready)
            val fixture = File(instrumentation.targetContext.cacheDir, "smoke-banana.jpg")
            instrumentation.context.assets.open("banana.jpg").use { source -> fixture.outputStream().use(source::copyTo) }
            scenario.onActivity {
                assertTrue(model.beginCapture())
                model.classify(Uri.fromFile(fixture), deleteAfter = true)
            }
            val scanDeadline = System.nanoTime() + 30_000_000_000
            while (model.state.value.busy && System.nanoTime() < scanDeadline) Thread.sleep(100)
            assertFalse("Pipeline should finish", model.state.value.busy)
            assertTrue(model.state.value.message, model.state.value.results.any { it.detection.fruit == "banana" })
            // The Compose test clock must render the asynchronous worker's result before capture.
            compose.waitForIdle()
            compose.onNodeWithText("Your scan").assertExists()
            assertEquals((1..model.state.value.results.size).toList(), model.state.value.results.map { it.id })
            compose.onNodeWithText("#1").assertExists()
            compose.onRoot().captureToImage().asAndroidBitmap().useBitmap { screenshot ->
                File(instrumentation.targetContext.filesDir, "smoke-result.png").outputStream().use {
                    screenshot.compress(Bitmap.CompressFormat.PNG, 100, it)
                }
            }
            Log.i("FruitLensTest", "pipelineProcessingMs=${model.state.value.processingMs}")
        }
    }

    @Test fun multipleFruitIdsMatchOverlayAndCards() {
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            lateinit var model: ScannerViewModel
            scenario.onActivity { model = ViewModelProvider(it)[ScannerViewModel::class.java] }
            compose.waitUntil(timeoutMillis = 30_000) { model.state.value.ready }
            val file = File(instrumentation.targetContext.cacheDir, "two-bananas.png")
            instrumentation.context.assets.open("two-bananas.png").use { input -> file.outputStream().use(input::copyTo) }
            scenario.onActivity { assertTrue(model.beginCapture()); model.classify(Uri.fromFile(file), deleteAfter = true) }
            compose.waitUntil(timeoutMillis = 30_000) { !model.state.value.busy }
            val results = model.state.value.results
            assertTrue("Should find both separate banana regions", results.size >= 2)
            assertEquals((1..results.size).toList(), results.map { it.id })
            compose.onNodeWithText("#1").assertExists()
            compose.onNodeWithText("#2").assertExists()
            val description = results.joinToString("; ") { "Fruit #${it.id}, ${it.detection.fruit}, ${FreshnessDisplay.assessment(it)}" }
            compose.onNodeWithContentDescription(description).assertExists()
            compose.onRoot().captureToImage().asAndroidBitmap().useBitmap { screenshot ->
                File(instrumentation.targetContext.filesDir, "multi-fruit-result.png").outputStream().use {
                    screenshot.compress(Bitmap.CompressFormat.PNG, 100, it)
                }
            }
        }
    }

    @Test fun cameraCaptureDecodesWithoutBlockingTheUi() {
        ParcelFileDescriptor.AutoCloseInputStream(instrumentation.uiAutomation.executeShellCommand(
            "pm grant ${instrumentation.targetContext.packageName} android.permission.CAMERA")).use { it.readBytes() }
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            lateinit var model: ScannerViewModel
            scenario.onActivity { model = ViewModelProvider(it)[ScannerViewModel::class.java] }
            compose.waitUntil(timeoutMillis = 30_000) {
                try { compose.onNodeWithText("Check this fruit").assertIsEnabled(); true }
                catch (_: AssertionError) { false }
            }
            compose.onNodeWithText("Check this fruit").performClick()
            val scanDeadline = System.nanoTime() + 30_000_000_000
            while ((model.state.value.busy || model.state.value.photo == null) && System.nanoTime() < scanDeadline) Thread.sleep(100)
            assertNotNull("Capture should produce a decoded photo: ${model.state.value.message}", model.state.value.photo)
            assertFalse(model.state.value.busy)
            Log.i("FruitLensTest", "cameraCaptureDecoded=${model.state.value.photo!!.width}x${model.state.value.photo!!.height}")
        }
    }
}

private inline fun Bitmap.useBitmap(block: (Bitmap) -> Unit) { try { block(this) } finally { recycle() } }
