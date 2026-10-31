package com.fruitlens.app

import android.Manifest
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import java.io.File
import java.util.Locale

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { FruitLensTheme { ScannerScreen() } }
    }
}

@Composable
private fun FruitLensTheme(content: @Composable () -> Unit) {
    val colors = if (isSystemInDarkTheme()) darkColorScheme(
        primary = Color(0xFF9AD5B3), onPrimary = Color(0xFF003821),
        background = Color(0xFF111812), surface = Color(0xFF18221B),
        onBackground = Color(0xFFE5EAE2), onSurface = Color(0xFFE5EAE2),
        onSurfaceVariant = Color(0xFFBECABC), surfaceVariant = Color(0xFF28382D),
    ) else lightColorScheme(
        primary = Color(0xFF185C3B), onPrimary = Color.White,
        background = Color(0xFFF6F7F0), surface = Color(0xFFFFFFFF),
        onBackground = Color(0xFF17251C), onSurface = Color(0xFF17251C),
        onSurfaceVariant = Color(0xFF46574A), surfaceVariant = Color(0xFFE6EDE4),
    )
    MaterialTheme(colorScheme = colors, content = content)
}

@Composable
private fun ScannerScreen(model: ScannerViewModel = viewModel()) {
    val state by model.state.collectAsStateWithLifecycle()
    val context = LocalContext.current
    var permission by remember {
        mutableStateOf(ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
    }
    val permissionLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { permission = it }
    var capture by remember { mutableStateOf<ImageCapture?>(null) }
    var cameraError by remember { mutableStateOf<String?>(null) }
    val photoPicker = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null && model.beginCapture()) model.classify(uri)
    }

    Surface(color = MaterialTheme.colorScheme.background, modifier = Modifier.fillMaxSize()) {
        BoxWithConstraints(Modifier.fillMaxSize().windowInsetsPadding(WindowInsets.safeDrawing)) {
            val wide = maxWidth > 700.dp
            val gutter = if (wide) 32.dp else 24.dp
            Column(
                Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = gutter, vertical = 24.dp),
                verticalArrangement = Arrangement.spacedBy(24.dp),
            ) {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("FRUIT LENS", style = MaterialTheme.typography.labelLarge,
                        color = MaterialTheme.colorScheme.primary, fontWeight = FontWeight.Bold)
                    Text(if (state.photo != null) "Your scan" else "A closer look at your fruit.",
                        style = if (state.photo != null) MaterialTheme.typography.headlineMedium else MaterialTheme.typography.headlineLarge,
                        fontWeight = FontWeight.SemiBold)
                    Text(if (state.photo != null) "A visual check, entirely on your device." else "One fruit. Good light. A clearer view.",
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                }

                Box(
                    Modifier.fillMaxWidth().heightIn(min = 180.dp, max = 440.dp)
                        .aspectRatio(if (state.photo != null || wide) 1.5f else 1f)
                        .clip(RoundedCornerShape(28.dp)).background(MaterialTheme.colorScheme.surfaceVariant),
                    contentAlignment = Alignment.Center,
                ) {
                    when {
                        state.photo != null -> {
                            val photo = state.photo!!
                            Image(photo.asImageBitmap(), contentDescription = "Captured fruit photo with detected regions",
                                contentScale = ContentScale.Fit, modifier = Modifier.fillMaxSize())
                            Canvas(Modifier.fillMaxSize()) {
                                val scale = minOf(size.width / photo.width, size.height / photo.height)
                                val width = photo.width * scale
                                val height = photo.height * scale
                                val offset = Offset((size.width - width) / 2, (size.height - height) / 2)
                                state.results.forEach {
                                    val b = it.detection.bounds
                                    drawRect(Color(0xFF9AD5B3), offset + Offset(b.left * width, b.top * height),
                                        Size(b.width() * width, b.height() * height), style = Stroke(3.dp.toPx()))
                                }
                            }
                        }
                        !permission -> Column(Modifier.padding(24.dp), horizontalAlignment = Alignment.CenterHorizontally,
                            verticalArrangement = Arrangement.spacedBy(16.dp)) {
                            Text("Bring your fruit into view", style = MaterialTheme.typography.titleLarge)
                            Text("Camera access is used only while you scan.")
                            Button(onClick = { permissionLauncher.launch(Manifest.permission.CAMERA) }) { Text("Enable camera") }
                        }
                        cameraError != null -> Text(cameraError!!, Modifier.padding(24.dp))
                        else -> CameraPane(Modifier.fillMaxSize(),
                            onReady = { capture = it }, onError = { capture = null; cameraError = it })
                    }
                    if (permission && cameraError == null && state.photo == null) Surface(
                        modifier = Modifier.align(Alignment.BottomCenter).padding(16.dp),
                        color = Color(0xDD17251C), contentColor = Color.White,
                        shape = RoundedCornerShape(16.dp),
                    ) { Text("Keep one fruit centered and fill the view", Modifier.padding(12.dp),
                        style = MaterialTheme.typography.bodyMedium) }
                }

                ScanResults(state)

                Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    Button(
                        onClick = {
                            if (state.photo != null) {
                                model.resetResult()
                                return@Button
                            }
                            val camera = capture
                            if (camera != null && model.beginCapture()) {
                                val photo = try { File.createTempFile("fruit-", ".jpg", context.cacheDir) }
                                catch (_: Exception) {
                                    model.reportError("There isn't enough temporary storage for a photo. Free some space and try again.")
                                    return@Button
                                }
                                try {
                                    camera.takePicture(ImageCapture.OutputFileOptions.Builder(photo).build(),
                                        ContextCompat.getMainExecutor(context), object : ImageCapture.OnImageSavedCallback {
                                            override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                                                model.classify(Uri.fromFile(photo), deleteAfter = true)
                                            }
                                            override fun onError(exception: ImageCaptureException) {
                                                photo.delete()
                                                model.reportError("Photo capture failed. Try again.")
                                            }
                                        })
                                } catch (_: Exception) {
                                    photo.delete()
                                    model.reportError("Photo capture failed. Try again.")
                                }
                            }
                        },
                        enabled = state.ready && !state.busy && (state.photo != null || (capture != null && permission && cameraError == null)),
                        modifier = Modifier.fillMaxWidth().heightIn(min = 56.dp),
                    ) { Text(if (state.busy) "Assessing photo…" else if (state.photo != null) "Scan another fruit" else "Check this fruit") }
                    OutlinedButton(onClick = { photoPicker.launch("image/*") }, enabled = state.ready && !state.busy,
                        modifier = Modifier.fillMaxWidth().heightIn(min = 52.dp)) { Text("Choose a photo") }
                }

                when {
                    state.loading -> Row(horizontalArrangement = Arrangement.spacedBy(12.dp), verticalAlignment = Alignment.CenterVertically) {
                        CircularProgressIndicator(Modifier.size(24.dp))
                        Text("Preparing the classifier…")
                    }
                    state.busy -> LinearProgressIndicator(Modifier.fillMaxWidth())
                }
                state.message?.let { message ->
                    InfoCard(if (state.ready) "Try another photo" else "Waiting for the classifier", message)
                }
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("Made for a quick visual check", style = MaterialTheme.typography.titleMedium)
                    Text("A photo shows surface appearance. It cannot confirm food safety or internal freshness.",
                        style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    if (state.modelId.isNotBlank()) Text("Classifier: ${state.modelId}",
                        style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
        }
    }
}

@Composable
private fun ScanResults(state: ScannerState) {
    if (state.results.isNotEmpty()) Text("${state.results.size} fruit found · ${state.processingMs} ms processing",
        style = MaterialTheme.typography.labelLarge)
    state.results.forEach { detected ->
        val result = detected.classification
        val agreement = result.label.fruit == detected.detection.fruit
        val heading = if (agreement) result.label.displayName
            else "${detected.detection.fruit.replaceFirstChar { it.uppercase() }} · Assessment uncertain"
        val detail = when {
            result.accepted -> "Based on the visible surface in this photo."
            agreement -> "Experimental suggestion. This classifier has not been independently validated for your camera."
            else -> "The fruit type is uncertain. Try a closer view with brighter, even lighting."
        }
        InfoCard(heading, "$detail\n\nModel score: ${String.format(Locale.ROOT, "%.3f", result.score)}")
    }
}

@Composable
private fun InfoCard(title: String, detail: String) {
    Surface(shape = RoundedCornerShape(24.dp), color = MaterialTheme.colorScheme.surface,
        modifier = Modifier.fillMaxWidth()) {
        Column(Modifier.padding(24.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Text(title, style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.SemiBold)
            Text(detail, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

@Composable
private fun CameraPane(modifier: Modifier, onReady: (ImageCapture?) -> Unit, onError: (String) -> Unit) {
    val context = LocalContext.current
    val owner = LocalLifecycleOwner.current
    val previewView = remember { PreviewView(context).apply { implementationMode = PreviewView.ImplementationMode.COMPATIBLE } }
    val ready by rememberUpdatedState(onReady)
    val error by rememberUpdatedState(onError)
    AndroidView(factory = { previewView }, modifier = modifier)
    DisposableEffect(owner, previewView) {
        var disposed = false
        var provider: ProcessCameraProvider? = null
        val preview = Preview.Builder().build().also { it.setSurfaceProvider(previewView.surfaceProvider) }
        val capture = ImageCapture.Builder().setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY).build()
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            if (!disposed) {
                try {
                    provider = future.get()
                    provider!!.bindToLifecycle(owner, CameraSelector.DEFAULT_BACK_CAMERA, preview, capture)
                    ready(capture)
                } catch (_: Exception) { error("Camera unavailable. You can still choose a photo.") }
            }
        }, ContextCompat.getMainExecutor(context))
        onDispose {
            disposed = true
            ready(null)
            provider?.unbind(preview, capture)
        }
    }
}
