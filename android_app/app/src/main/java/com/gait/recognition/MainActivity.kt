package com.gait.recognition

import ai.onnxruntime.OrtEnvironment
import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
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

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: ImageView
    private lateinit var silhouettePreview: ImageView
    private lateinit var statusText: TextView
    private lateinit var top3Text: TextView
    private lateinit var btnToggle: Button
    private lateinit var btnReset: Button
    private lateinit var btnVideo: Button

    private val ortEnv = OrtEnvironment.getEnvironment()
    private var yolo: YoloProcessor? = null
    private var gaitEngine: GaitEngine? = null

    private val silBuffer = ArrayDeque<FloatArray>(60)
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
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        if (uri != null) analyzeVideoUri(uri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        overlayView = findViewById(R.id.overlayView)
        silhouettePreview = findViewById(R.id.silhouettePreview)
        statusText = findViewById(R.id.statusText)
        top3Text = findViewById(R.id.top3Text)
        btnToggle = findViewById(R.id.btnToggle)
        btnReset = findViewById(R.id.btnReset)
        btnVideo = findViewById(R.id.btnVideo)

        btnToggle.setOnClickListener {
            isAnalyzing = !isAnalyzing
            btnToggle.text = if (isAnalyzing) "Detener" else "Iniciar"
            if (!isAnalyzing && votes.isNotEmpty()) {
                showFinalConclusion()
            }
        }

        btnReset.setOnClickListener { resetSession() }
        btnVideo.setOnClickListener { videoPicker.launch("video/*") }

        thread {
            try {
                yolo = YoloProcessor(this, ortEnv)
                gaitEngine = GaitEngine(this, ortEnv)
                runOnUiThread {
                    statusText.text = "Modelos listos (31 sujetos offline). Presiona Iniciar."
                    btnToggle.isEnabled = true
                }
            } catch (e: Exception) {
                runOnUiThread { statusText.text = "Error: ${e.message}" }
            }
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            camPermLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun resetSession() {
        synchronized(silBuffer) { silBuffer.clear() }
        votes.clear()
        frameCounter = 0
        currentLabel = "Analizando..."
        currentColor = Color.YELLOW
        statusText.text = "Caminata reiniciada"
        top3Text.text = "Top-3: --"
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }
            val analyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
            analyzer.setAnalyzer(analysisExecutor) { imgProxy ->
                processImageProxy(imgProxy)
            }
            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analyzer)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun processImageProxy(proxy: ImageProxy) {
        if (!isAnalyzing || yolo == null || gaitEngine == null) {
            proxy.close()
            return
        }
        val bmp = proxyToBitmap(proxy)
        proxy.close()
        if (bmp != null) {
            runFramePipeline(bmp)
        }
    }

    private fun analyzeVideoUri(uri: Uri) {
        resetSession()
        isAnalyzing = false
        btnToggle.text = "Iniciar"
        statusText.text = "Procesando video seleccionado..."
        thread {
            val retriever = MediaMetadataRetriever()
            retriever.setDataSource(this, uri)
            val durMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
            val stepMs = 66L // ~15 fps de muestreo
            var t = 0L
            while (t < durMs) {
                val bmp = retriever.getFrameAtTime(t * 1000L, MediaMetadataRetriever.OPTION_CLOSEST)
                if (bmp != null) runFramePipeline(bmp)
                t += stepMs
            }
            retriever.release()
            runOnUiThread { showFinalConclusion() }
        }
    }

    private fun runFramePipeline(bmp: Bitmap) {
        val res = yolo!!.processFrame(bmp)
        if (res.alignedSil != null) {
            synchronized(silBuffer) {
                if (silBuffer.size >= 45) silBuffer.removeFirst()
                silBuffer.addLast(res.alignedSil)
            }
        }
        frameCounter++

        var top3Str = ""
        if (frameCounter % 6 == 0) {
            val seq = synchronized(silBuffer) { silBuffer.toList() }
            if (seq.size >= 10) {
                val ranked = gaitEngine!!.identify(seq)
                if (ranked.isNotEmpty()) {
                    val t1 = ranked[0]
                    val t2Score = if (ranked.size > 1) ranked[1].score else 0f
                    val margin = t1.score - t2Score
                    if (t1.score >= 0.945f && margin >= 0.010f) {
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

        val overlay = YoloProcessor.drawOverlay(bmp.width, bmp.height, res.box, res.keypoints, currentLabel, currentColor)
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
        val winner = known.maxByOrNull { it.value }
        val finalId = if (winner != null && total > 0 && winner.value.toFloat() / total >= 0.25f) {
            winner.key
        } else {
            "Desconocido"
        }
        AlertDialog.Builder(this)
            .setTitle("Conclusión Final")
            .setMessage("Identidad: $finalId\n\nVotos: $votes")
            .setPositiveButton("OK", null)
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

