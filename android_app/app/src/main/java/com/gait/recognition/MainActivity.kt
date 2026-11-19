package com.gait.recognition

import ai.onnxruntime.OrtEnvironment
import android.Manifest
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.content.res.Configuration
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.view.Surface
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.util.ArrayDeque
import java.util.concurrent.Executors
import kotlin.concurrent.thread
import kotlin.math.abs

enum class AppMode {
    CAMERA,
    VIDEO
}

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: ImageView
    private lateinit var silhouettePreview: ImageView
    private lateinit var modeBadge: TextView
    private lateinit var videoProgressLayout: LinearLayout
    private lateinit var videoProgressText: TextView
    private lateinit var videoProgressBar: ProgressBar
    private lateinit var statusText: TextView
    private lateinit var top3Text: TextView
    private lateinit var btnToggle: Button
    private lateinit var btnReset: Button
    private lateinit var btnVideo: Button

    private val ortEnv = OrtEnvironment.getEnvironment()
    private var yolo: YoloProcessor? = null
    private var gaitEngine: GaitEngine? = null

    private var cameraProvider: ProcessCameraProvider? = null
    private var previewUseCase: Preview? = null
    private var analysisUseCase: ImageAnalysis? = null
    private var currentMode = AppMode.CAMERA

    private val silBuffer = ArrayDeque<FloatArray>(60)
    private val kptBuffer = ArrayDeque<Array<FloatArray>>(60)
    private val motionHistory = ArrayDeque<Pair<Float, Float>>(5)
    private val votes = HashMap<String, Int>()
    private var isAnalyzing = false
    private var frameCounter = 0
    private var currentLabel = "Analizando..."
    private var currentColor = Color.YELLOW

    private val analysisExecutor = Executors.newSingleThreadExecutor()

    private val camPermLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera() else statusText.text = "Permiso de cámara denegado"
    }

    private val videoPicker = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        if (uri != null) analyzeVideoUri(uri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        overlayView = findViewById(R.id.overlayView)
        silhouettePreview = findViewById(R.id.silhouettePreview)
        modeBadge = findViewById(R.id.modeBadge)
        videoProgressLayout = findViewById(R.id.videoProgressLayout)
        videoProgressText = findViewById(R.id.videoProgressText)
        videoProgressBar = findViewById(R.id.videoProgressBar)
        statusText = findViewById(R.id.statusText)
        top3Text = findViewById(R.id.top3Text)
        btnToggle = findViewById(R.id.btnToggle)
        btnReset = findViewById(R.id.btnReset)
        btnVideo = findViewById(R.id.btnVideo)

        btnToggle.setOnClickListener {
            if (currentMode == AppMode.VIDEO) {
                switchToCameraMode()
            } else {
                isAnalyzing = !isAnalyzing
                updateToggleButtonState()
                if (!isAnalyzing && votes.isNotEmpty()) {
                    showFinalConclusion()
                }
            }
        }

        btnReset.setOnClickListener { resetSession() }
        btnVideo.setOnClickListener {
            if (yolo == null || gaitEngine == null) {
                statusText.text = "Espera a que terminen de cargar los modelos..."
            } else {
                videoPicker.launch(arrayOf("video/*"))
            }
        }

        thread {
            try {
                yolo = YoloProcessor(this, ortEnv)
                gaitEngine = GaitEngine(this, ortEnv)
                runOnUiThread {
                    statusText.text = "Sistema biométrico listo. Presiona Iniciar."
                    btnToggle.isEnabled = true
                    updateToggleButtonState()
                }
            } catch (e: Exception) {
                runOnUiThread { statusText.text = "Error cargando modelos: ${e.message}" }
            }
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            camPermLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun getDisplayRotation(): Int {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            display?.rotation ?: Surface.ROTATION_0
        } else {
            @Suppress("DEPRECATION")
            windowManager.defaultDisplay.rotation
        }
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        val rot = getDisplayRotation()
        previewUseCase?.targetRotation = rot
        analysisUseCase?.targetRotation = rot
    }

    private fun resetSession() {
        synchronized(silBuffer) {
            silBuffer.clear()
            kptBuffer.clear()
            motionHistory.clear()
        }
        votes.clear()
        frameCounter = 0
        currentLabel = "Analizando..."
        currentColor = Color.YELLOW
        overlayView.setImageDrawable(null)
        silhouettePreview.setImageDrawable(null)
        statusText.text = "Caminata reiniciada"
        top3Text.text = "Top-3: --"
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            cameraProvider = providerFuture.get()
            bindCameraUseCases()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCameraUseCases() {
        val provider = cameraProvider ?: return
        val rotation = getDisplayRotation()

        previewUseCase = Preview.Builder()
            .setTargetRotation(rotation)
            .build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

        analysisUseCase = ImageAnalysis.Builder()
            .setTargetRotation(rotation)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build().also {
                it.setAnalyzer(analysisExecutor) { imgProxy ->
                    processImageProxy(imgProxy)
                }
            }

        try {
            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, previewUseCase, analysisUseCase)
        } catch (e: Exception) {
            statusText.text = "Error al vincular cámara: ${e.message}"
        }
    }

    private fun updateToggleButtonState() {
        when (currentMode) {
            AppMode.VIDEO -> {
                btnToggle.text = "↺ Volver a Cámara"
                btnToggle.backgroundTintList = ColorStateList.valueOf(Color.parseColor("#10B981"))
                btnToggle.setTextColor(Color.WHITE)
            }
            AppMode.CAMERA -> {
                if (isAnalyzing) {
                    btnToggle.text = "⏹ Detener"
                    btnToggle.backgroundTintList = ColorStateList.valueOf(Color.parseColor("#F43F5E"))
                    btnToggle.setTextColor(Color.WHITE)
                } else {
                    btnToggle.text = "▶ Iniciar"
                    btnToggle.backgroundTintList = ColorStateList.valueOf(Color.parseColor("#38BDF8"))
                    btnToggle.setTextColor(Color.parseColor("#030712"))
                }
            }
        }
    }

    private fun switchToCameraMode() {
        currentMode = AppMode.CAMERA
        videoProgressLayout.visibility = View.GONE
        previewView.visibility = View.VISIBLE
        modeBadge.text = "● MODO CÁMARA EN VIVO"
        modeBadge.setTextColor(Color.parseColor("#38BDF8"))
        isAnalyzing = false
        updateToggleButtonState()
        resetSession()
        bindCameraUseCases()
    }

    private fun processImageProxy(proxy: ImageProxy) {
        if (!isAnalyzing || currentMode != AppMode.CAMERA || yolo == null || gaitEngine == null) {
            proxy.close()
            return
        }
        val bmp = proxyToBitmap(proxy)
        proxy.close()
        if (bmp != null) {
            runFramePipeline(bmp, drawBg = false)
        }
    }

    private fun analyzeVideoUri(uri: Uri) {
        resetSession()
        isAnalyzing = false
        currentMode = AppMode.VIDEO

        // Desactivar cámara y ocultarla para que no se superponga
        cameraProvider?.unbindAll()
        previewView.visibility = View.GONE

        // Mostrar controles de progreso de video
        videoProgressLayout.visibility = View.VISIBLE
        videoProgressBar.progress = 0
        videoProgressText.text = "Cargando archivo de video..."
        modeBadge.text = "● MODO ANÁLISIS DE VIDEO"
        modeBadge.setTextColor(Color.parseColor("#FFA657"))
        updateToggleButtonState()
        statusText.text = "Preparando video..."

        thread {
            val retriever = MediaMetadataRetriever()
            try {
                val tempFile = java.io.File(cacheDir, "picked_video.mp4")
                contentResolver.openInputStream(uri)?.use { input ->
                    tempFile.outputStream().use { output -> input.copyTo(output) }
                } ?: throw IllegalStateException("No se pudo leer el archivo seleccionado")

                retriever.setDataSource(tempFile.absolutePath)
                val durMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
                val stepMs = 80L // ~12.5 fps de muestreo
                var t = 0L

                val totalSteps = if (durMs > 0) (durMs / stepMs).toInt().coerceAtLeast(1) else 1
                var stepIndex = 0

                while (t < durMs) {
                    val bmp = retriever.getFrameAtTime(t * 1000L, MediaMetadataRetriever.OPTION_CLOSEST)
                    if (bmp != null) {
                        runFramePipeline(bmp, drawBg = true)
                    }

                    stepIndex++
                    val progress = ((t.toFloat() / durMs.toFloat()) * 100).toInt().coerceIn(0, 100)
                    val secCurrent = String.format("%.1f", t / 1000f)
                    val secTotal = String.format("%.1f", durMs / 1000f)

                    runOnUiThread {
                        videoProgressBar.progress = progress
                        videoProgressText.text = "Analizando video: $progress% ($secCurrent s / $secTotal s)"
                    }

                    t += stepMs
                }

                runOnUiThread {
                    videoProgressBar.progress = 100
                    videoProgressText.text = "Análisis completado (100%)"
                    showFinalConclusion()
                }
            } catch (e: Exception) {
                runOnUiThread {
                    statusText.text = "Error abriendo video: ${e.message}"
                    videoProgressText.text = "Error al procesar video"
                }
            } finally {
                try { retriever.release() } catch (_: Exception) {}
            }
        }
    }

    private fun runFramePipeline(bmp: Bitmap, drawBg: Boolean = false) {
        val res = yolo!!.processFrame(bmp)
        var isWalkingNow = false
        if (res.box != null && res.alignedSil != null) {
            val cx = (res.box.left + res.box.right) / 2f
            val bh = res.box.bottom - res.box.top
            synchronized(silBuffer) {
                if (motionHistory.size >= 5) motionHistory.removeFirst()
                motionHistory.addLast(cx to bh)
                val first = motionHistory.first()
                val dcx = abs(cx - first.first)
                val dbh = abs(bh - first.second)
                isWalkingNow = (motionHistory.size < 3) || (dcx >= 0.015f) || (dbh >= 0.010f)
                if (isWalkingNow && bh >= 0.20f) {
                    if (silBuffer.size >= 45) silBuffer.removeFirst()
                    silBuffer.addLast(res.alignedSil)
                    if (res.keypoints != null) {
                        if (kptBuffer.size >= 45) kptBuffer.removeFirst()
                        kptBuffer.addLast(res.keypoints)
                    }
                }
            }
        }
        frameCounter++

        var top3Str = ""
        if (isWalkingNow && frameCounter % 6 == 0) {
            val (seq, kSeq) = synchronized(silBuffer) { silBuffer.toList() to kptBuffer.toList() }
            if (seq.size >= 10) {
                val ranked = gaitEngine!!.identify(seq, kSeq, bmp.width, bmp.height, alphaSil = 0.65f)
                if (ranked.isNotEmpty()) {
                    val t1 = ranked[0]
                    val t2Score = if (ranked.size > 1) ranked[1].score else 0f
                    val margin = t1.score - t2Score
                    if (t1.score >= 0.540f && margin >= 0.025f) {
                        currentLabel = "${t1.subject} (${"%.2f".format(t1.score)})"
                        currentColor = Color.rgb(0, 230, 140)
                        votes[t1.subject] = (votes[t1.subject] ?: 0) + 1
                    } else {
                        currentLabel = "Desconocido (${"%.2f".format(t1.score)})"
                        currentColor = Color.rgb(255, 165, 0)
                        votes["Desconocido"] = (votes["Desconocido"] ?: 0) + 1
                    }
                    top3Str = ranked.take(3).joinToString(" | ") { "${it.subject}: ${"%.3f".format(it.score)}" }
                }
            }
        }

        val overlay = YoloProcessor.drawOverlay(
            bmp.width, bmp.height, res.box, res.keypoints, currentLabel, currentColor,
            bgBitmap = if (drawBg) bmp else null
        )
        runOnUiThread {
            overlayView.setImageBitmap(overlay)
            if (res.silPreview != null) silhouettePreview.setImageBitmap(res.silPreview)
            statusText.text = currentLabel
            if (top3Str.isNotEmpty()) top3Text.text = "Top-3: $top3Str"
        }
    }

    private fun showFinalConclusion() {
        val total = votes.values.sum()
        val known = votes.filterKeys { it != "Desconocido" }
        val knownTotal = known.values.sum()
        val winner = known.maxByOrNull { it.value }
        val (finalId, confPct) = if (
            winner != null && total > 0 && knownTotal > 0 &&
            (winner.value.toFloat() / total >= 0.12f) &&
            (winner.value.toFloat() / knownTotal >= 0.50f)
        ) {
            winner.key to ((winner.value.toFloat() / total) * 100).toInt()
        } else {
            "Desconocido" to (if (total > 0) (((votes["Desconocido"] ?: 0).toFloat() / total) * 100).toInt() else 0)
        }

        val topVotes = votes.entries.sortedByDescending { it.value }.take(4).joinToString("\n") { (k, v) ->
            "  • $k: $v votos (${if (total > 0) ((v.toFloat() / total) * 100).toInt() else 0}%)"
        }

        AlertDialog.Builder(this)
            .setTitle("🎯 Conclusión Biométrica")
            .setMessage("Sujeto Identificado:\n$finalId ($confPct% de consenso)\n\nTotal de cuadros evaluados: $total\n\nDistribución de Votos:\n$topVotes")
            .setPositiveButton("Aceptar", null)
            .show()
    }

    private fun proxyToBitmap(proxy: ImageProxy): Bitmap? {
        val bmp = proxy.toBitmap()
        val rot = proxy.imageInfo.rotationDegrees
        if (rot == 0) return bmp
        val m = Matrix().apply { postRotate(rot.toFloat()) }
        return Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
    }

    override fun onDestroy() {
        super.onDestroy()
        analysisExecutor.shutdown()
    }
}
