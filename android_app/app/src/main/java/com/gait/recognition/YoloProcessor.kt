package com.gait.recognition

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import java.nio.FloatBuffer
import kotlin.math.acos
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

data class FrameResult(
    val box: RectF?,
    val alignedSil: FloatArray?,       // 64 * 44 en [0, 1]
    val silPreview: Bitmap?,           // 44x64 para mostrar en UI
    val keypoints: Array<FloatArray>?  // 17 x [x_norm, y_norm, conf] en coords [0..1] de imagen
)

class YoloProcessor(context: Context, private val env: OrtEnvironment) {
    private val segSession: OrtSession
    private val poseSession: OrtSession
    private val inputSize = 512

    init {
        val segBytes = context.assets.open("yolov8n-seg-512.onnx").readBytes()
        val poseBytes = context.assets.open("yolov8n-pose-512.onnx").readBytes()
        segSession = env.createSession(segBytes)
        poseSession = env.createSession(poseBytes)
    }

    fun processFrame(bitmap: Bitmap): FrameResult {
        val (letterbox, scale, padX, padY) = letterboxBitmap(bitmap, inputSize)
        val inputBuffer = bitmapToFloatBuffer(letterbox)
        val tensor = OnnxTensor.createTensor(env, inputBuffer, longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong()))

        var bestBox: RectF? = null
        var alignedSil: FloatArray? = null
        var silPreview: Bitmap? = null
        var keypoints: Array<FloatArray>? = null

        tensor.use { t ->
            val segOut = segSession.run(mapOf(segSession.inputNames.first() to t))
            segOut.use { res ->
                @Suppress("UNCHECKED_CAST")
                val dets = (res[0].value as Array<Array<FloatArray>>)[0] // [116, 5376]
                @Suppress("UNCHECKED_CAST")
                val protos = (res[1].value as Array<Array<Array<FloatArray>>>)[0] // [32, 128, 128]
                val segParsed = parseSeg(dets, protos, bitmap.width, bitmap.height, scale, padX, padY)
                if (segParsed != null) {
                    bestBox = segParsed.first
                    alignedSil = segParsed.second
                    silPreview = segParsed.third
                }
            }

            val poseOut = poseSession.run(mapOf(poseSession.inputNames.first() to t))
            poseOut.use { res ->
                @Suppress("UNCHECKED_CAST")
                val dets = (res[0].value as Array<Array<FloatArray>>)[0] // [56, 5376]
                keypoints = parsePose(dets, bitmap.width, bitmap.height, scale, padX, padY)
            }
        }

        return FrameResult(bestBox, alignedSil, silPreview, keypoints)
    }

    private fun parseSeg(
        dets: Array<FloatArray>,
        protos: Array<Array<FloatArray>>,
        origW: Int,
        origH: Int,
        scale: Float,
        padX: Float,
        padY: Float
    ): Triple<RectF, FloatArray, Bitmap>? {
        val numAnchors = dets[0].size
        var bestIdx = -1
        var bestHeight = 0f

        for (i in 0 until numAnchors) {
            val conf = dets[4][i] // clase 0 = person
            if (conf < 0.45f) continue
            val h = dets[3][i]
            if (h > bestHeight) {
                bestHeight = h
                bestIdx = i
            }
        }
        if (bestIdx < 0) return null

        val cx = dets[0][bestIdx]
        val cy = dets[1][bestIdx]
        val w = dets[2][bestIdx]
        val h = dets[3][bestIdx]

        val x1 = ((cx - w / 2f) - padX) / scale
        val y1 = ((cy - h / 2f) - padY) / scale
        val x2 = ((cx + w / 2f) - padX) / scale
        val y2 = ((cy + h / 2f) - padY) / scale

        val box = RectF(
            (x1 / origW).coerceIn(0f, 1f),
            (y1 / origH).coerceIn(0f, 1f),
            (x2 / origW).coerceIn(0f, 1f),
            (y2 / origH).coerceIn(0f, 1f)
        )

        // Coordenadas en espacio proto (128x128 = 512 / 4)
        val px1 = (((cx - w / 2f) / 4f).toInt()).coerceIn(0, 127)
        val py1 = (((cy - h / 2f) / 4f).toInt()).coerceIn(0, 127)
        val px2 = (((cx + w / 2f) / 4f).toInt()).coerceIn(px1 + 1, 128)
        val py2 = (((cy + h / 2f) / 4f).toInt()).coerceIn(py1 + 1, 128)

        val coeffs = FloatArray(32) { k -> dets[84 + k][bestIdx] }
        val cropW = px2 - px1
        val cropH = py2 - py1
        val mask = BooleanArray(cropW * cropH)
        var fgCount = 0
        var minMy = cropH
        var maxMy = -1
        var minMx = cropW
        var maxMx = -1

        for (my in 0 until cropH) {
            val py = py1 + my
            for (mx in 0 until cropW) {
                val px = px1 + mx
                var sum = 0f
                for (k in 0 until 32) {
                    sum += coeffs[k] * protos[k][py][px]
                }
                val prob = 1f / (1f + exp(-sum))
                if (prob > 0.5f) {
                    mask[my * cropW + mx] = true
                    fgCount++
                    if (my < minMy) minMy = my
                    if (my > maxMy) maxMy = my
                    if (mx < minMx) minMx = mx
                    if (mx > maxMx) maxMx = mx
                }
            }
        }
        if (fgCount < 20 || maxMy <= minMy || maxMx <= minMx) return null

        // Alinear a 64x44 estilo OpenGait
        val outH = 64
        val outW = 44
        val tightH = maxMy - minMy + 1
        val tightW = maxMx - minMx + 1
        val silScale = outH.toFloat() / tightH.toFloat()
        val newW = max(1, (tightW * silScale).roundToInt())

        val resized = BooleanArray(outH * newW)
        var sumFgX = 0
        var sumFgN = 0
        for (oy in 0 until outH) {
            val sy = minMy + min(tightH - 1, (oy / silScale).toInt())
            for (ox in 0 until newW) {
                val sx = minMx + min(tightW - 1, (ox / silScale).toInt())
                if (mask[sy * cropW + sx]) {
                    resized[oy * newW + ox] = true
                    sumFgX += ox
                    sumFgN++
                }
            }
        }
        val centerX = if (sumFgN > 0) (sumFgX.toFloat() / sumFgN).roundToInt() else newW / 2
        val xOffset = (outW / 2) - centerX

        val canvas = FloatArray(outH * outW)
        val preview = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888)
        for (oy in 0 until outH) {
            for (ox in 0 until outW) {
                val sx = ox - xOffset
                val fg = sx in 0 until newW && resized[oy * newW + sx]
                canvas[oy * outW + ox] = if (fg) 1f else 0f
                preview.setPixel(ox, oy, if (fg) Color.WHITE else Color.BLACK)
            }
        }

        return Triple(box, canvas, preview)
    }

    private fun parsePose(
        dets: Array<FloatArray>,
        origW: Int,
        origH: Int,
        scale: Float,
        padX: Float,
        padY: Float
    ): Array<FloatArray>? {
        val numAnchors = dets[0].size
        var bestIdx = -1
        var bestHeight = 0f
        for (i in 0 until numAnchors) {
            if (dets[4][i] < 0.45f) continue
            if (dets[3][i] > bestHeight) {
                bestHeight = dets[3][i]
                bestIdx = i
            }
        }
        if (bestIdx < 0) return null

        return Array(17) { k ->
            val kx = ((dets[5 + k * 3][bestIdx] - padX) / scale) / origW
            val ky = ((dets[5 + k * 3 + 1][bestIdx] - padY) / scale) / origH
            val kc = dets[5 + k * 3 + 2][bestIdx]
            floatArrayOf(kx.coerceIn(0f, 1f), ky.coerceIn(0f, 1f), kc)
        }
    }

    private data class LetterboxResult(val bmp: Bitmap, val scale: Float, val padX: Float, val padY: Float)

    private fun letterboxBitmap(src: Bitmap, target: Int): LetterboxResult {
        val scale = min(target.toFloat() / src.width, target.toFloat() / src.height)
        val newW = (src.width * scale).roundToInt()
        val newH = (src.height * scale).roundToInt()
        val padX = (target - newW) / 2f
        val padY = (target - newH) / 2f

        val out = Bitmap.createBitmap(target, target, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(out)
        canvas.drawColor(Color.rgb(114, 114, 114))
        val scaled = Bitmap.createScaledBitmap(src, newW, newH, true)
        canvas.drawBitmap(scaled, padX, padY, null)
        return LetterboxResult(out, scale, padX, padY)
    }

    private fun bitmapToFloatBuffer(bmp: Bitmap): FloatBuffer {
        val w = bmp.width
        val h = bmp.height
        val pixels = IntArray(w * h)
        bmp.getPixels(pixels, 0, w, 0, 0, w, h)
        val fb = FloatBuffer.allocate(3 * w * h)
        val area = w * h
        for (i in 0 until area) {
            val p = pixels[i]
            fb.put(i, ((p shr 16) and 0xFF) / 255f)
            fb.put(area + i, ((p shr 8) and 0xFF) / 255f)
            fb.put(2 * area + i, (p and 0xFF) / 255f)
        }
        fb.rewind()
        return fb
    }

    companion object {
        val COCO_SKELETON = listOf(
            0 to 1, 0 to 2, 1 to 3, 2 to 4,
            5 to 6, 5 to 11, 6 to 12, 11 to 12,
            5 to 7, 7 to 9, 6 to 8, 8 to 10,
            11 to 13, 13 to 15, 12 to 14, 14 to 16
        )

        fun drawOverlay(width: Int, height: Int, box: RectF?, kpts: Array<FloatArray>?, label: String, color: Int): Bitmap {
            val bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(bmp)
            val boxPaint = Paint().apply {
                this.color = color
                style = Paint.Style.STROKE
                strokeWidth = 5f
            }
            val textPaint = Paint().apply {
                this.color = color
                textSize = 42f
                isFakeBoldText = true
            }
            if (box != null) {
                val r = RectF(box.left * width, box.top * height, box.right * width, box.bottom * height)
                canvas.drawRect(r, boxPaint)
                canvas.drawText(label, r.left, max(46f, r.top - 12f), textPaint)
            }
            if (kpts != null) {
                val linePaint = Paint().apply { strokeWidth = 6f; isAntiAlias = true }
                val ptPaint = Paint().apply { style = Paint.Style.FILL; isAntiAlias = true }
                for ((i, j) in COCO_SKELETON) {
                    if (kpts[i][2] >= 0.3f && kpts[j][2] >= 0.3f) {
                        linePaint.color = if (i % 2 == 1 && j % 2 == 1) Color.rgb(80, 230, 120) else Color.rgb(40, 165, 255)
                        canvas.drawLine(
                            kpts[i][0] * width, kpts[i][1] * height,
                            kpts[j][0] * width, kpts[j][1] * height, linePaint
                        )
                    }
                }
                for (k in 0 until 17) {
                    if (kpts[k][2] >= 0.3f) {
                        ptPaint.color = if (k % 2 == 1) Color.GREEN else Color.YELLOW
                        canvas.drawCircle(kpts[k][0] * width, kpts[k][1] * height, 8f, ptPaint)
                    }
                }
            }
            return bmp
        }
    }
}

