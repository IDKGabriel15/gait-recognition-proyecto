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
        var bestScore = 0f

        for (i in 0 until numAnchors) {
            val conf = dets[4][i] // clase 0 = person
            val h = dets[3][i]
            if (conf < 0.55f || h < 40f) continue
            val sc = conf * min(h, 400f)
            if (sc > bestScore) {
                bestScore = sc
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

        // 1. Calcular logits continuos en la grilla 128x128 alrededor de la caja (con margen de 2 px)
        val px1 = (((cx - w / 2f) / 4f).toInt() - 2).coerceIn(0, 127)
        val py1 = (((cy - h / 2f) / 4f).toInt() - 2).coerceIn(0, 127)
        val px2 = (((cx + w / 2f) / 4f).toInt() + 3).coerceIn(px1 + 1, 128)
        val py2 = (((cy + h / 2f) / 4f).toInt() + 3).coerceIn(py1 + 1, 128)

        val coeffs = FloatArray(32) { k -> dets[84 + k][bestIdx] }
        val protoW = px2 - px1
        val protoH = py2 - py1
        val logits128 = FloatArray(protoH * protoW)

        for (my in 0 until protoH) {
            val py = py1 + my
            for (mx in 0 until protoW) {
                val px = px1 + mx
                var sum = 0f
                for (k in 0 until 32) {
                    sum += coeffs[k] * protos[k][py][px]
                }
                logits128[my * protoW + mx] = sum
            }
        }

        // Función auxiliar: interpolación bilineal sobre logits128 en coordenadas de 512x512
        fun sampleLogit512(x512: Float, y512: Float): Float {
            // Mapear de 512x512 a 128x128 (align_corners=False estilo PyTorch/OpenCV)
            val gx = ((x512 + 0.5f) / 4f - 0.5f) - px1
            val gy = ((y512 + 0.5f) / 4f - 0.5f) - py1
            val x0 = gx.toInt().coerceIn(0, protoW - 1)
            val y0 = gy.toInt().coerceIn(0, protoH - 1)
            val x1p = (x0 + 1).coerceIn(0, protoW - 1)
            val y1p = (y0 + 1).coerceIn(0, protoH - 1)
            val wx = (gx - x0).coerceIn(0f, 1f)
            val wy = (gy - y0).coerceIn(0f, 1f)

            val v00 = logits128[y0 * protoW + x0]
            val v01 = logits128[y0 * protoW + x1p]
            val v10 = logits128[y1p * protoW + x0]
            val v11 = logits128[y1p * protoW + x1p]
            val top = v00 * (1f - wx) + v01 * wx
            val bot = v10 * (1f - wx) + v11 * wx
            return top * (1f - wy) + bot * wy
        }

        // 2. Evaluar máscara suavizada en espacio 512x512 dentro de la caja detectada
        val bx1 = (cx - w / 2f).roundToInt().coerceIn(0, inputSize - 1)
        val by1 = (cy - h / 2f).roundToInt().coerceIn(0, inputSize - 1)
        val bx2 = (cx + w / 2f).roundToInt().coerceIn(bx1 + 1, inputSize)
        val by2 = (cy + h / 2f).roundToInt().coerceIn(by1 + 1, inputSize)

        val cropW = bx2 - bx1
        val cropH = by2 - by1
        val mask512 = BooleanArray(cropH * cropW)
        var fgCount = 0
        var minMy = cropH
        var maxMy = -1
        var minMx = cropW
        var maxMx = -1

        for (my in 0 until cropH) {
            val y512 = (by1 + my).toFloat()
            for (mx in 0 until cropW) {
                val x512 = (bx1 + mx).toFloat()
                // logit > 0f equivale exactamente a sigmoid(logit) > 0.5f
                if (sampleLogit512(x512, y512) > 0f) {
                    mask512[my * cropW + mx] = true
                    fgCount++
                    if (my < minMy) minMy = my
                    if (my > maxMy) maxMy = my
                    if (mx < minMx) minMx = mx
                    if (mx > maxMx) maxMx = mx
                }
            }
        }
        if (fgCount < 80 || maxMy <= minMy || maxMx <= minMx) return null

        // 3. Alinear a 64x44 estilo OpenGait (para GaitBase)
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
                if (mask512[sy * cropW + sx]) {
                    resized[oy * newW + ox] = true
                    sumFgX += ox
                    sumFgN++
                }
            }
        }
        val centerX = if (sumFgN > 0) (sumFgX.toFloat() / sumFgN).roundToInt() else newW / 2
        val xOffset = (outW / 2) - centerX

        val canvas = FloatArray(outH * outW)
        for (oy in 0 until outH) {
            for (ox in 0 until outW) {
                val sx = ox - xOffset
                val fg = sx in 0 until newW && resized[oy * newW + sx]
                canvas[oy * outW + ox] = if (fg) 1f else 0f
            }
        }

        // 4. Generar silPreview en alta resolución (176x256, 4x) con bordes suaves para la UI
        val prevH = 256
        val prevW = 176
        val prevScale = prevH.toFloat() / tightH.toFloat()
        val prevNewW = max(1, (tightW * prevScale).roundToInt())
        val prevXOffset = (prevW / 2) - (centerX * 4)
        val preview = Bitmap.createBitmap(prevW, prevH, Bitmap.Config.ARGB_8888)
        val prevPixels = IntArray(prevW * prevH)
        for (oy in 0 until prevH) {
            val y512 = by1 + minMy + (oy / prevScale)
            for (ox in 0 until prevW) {
                val sx = ox - prevXOffset
                val fg = if (sx in 0 until prevNewW) {
                    val x512 = bx1 + minMx + (sx / prevScale)
                    sampleLogit512(x512, y512) > 0f
                } else {
                    false
                }
                prevPixels[oy * prevW + ox] = if (fg) Color.WHITE else Color.BLACK
            }
        }
        preview.setPixels(prevPixels, 0, prevW, 0, 0, prevW, prevH)

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
        var bestScore = 0f
        for (i in 0 until numAnchors) {
            val conf = dets[4][i]
            val h = dets[3][i]
            if (conf < 0.55f || h < 40f) continue
            val sc = conf * min(h, 400f)
            if (sc > bestScore) {
                bestScore = sc
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

        fun drawOverlay(
            width: Int,
            height: Int,
            box: RectF?,
            kpts: Array<FloatArray>?,
            label: String,
            color: Int,
            bgBitmap: Bitmap? = null
        ): Bitmap {
            val bmp = if (bgBitmap != null) {
                bgBitmap.copy(Bitmap.Config.ARGB_8888, true)
            } else {
                Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            }
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
                    if (kpts[i][2] >= 0.45f && kpts[j][2] >= 0.45f) {
                        linePaint.color = if (i % 2 == 1 && j % 2 == 1) Color.rgb(80, 230, 120) else Color.rgb(40, 165, 255)
                        canvas.drawLine(
                            kpts[i][0] * width, kpts[i][1] * height,
                            kpts[j][0] * width, kpts[j][1] * height, linePaint
                        )
                    }
                }
                for (k in 0 until 17) {
                    if (kpts[k][2] >= 0.45f) {
                        ptPaint.color = if (k % 2 == 1) Color.GREEN else Color.YELLOW
                        canvas.drawCircle(kpts[k][0] * width, kpts[k][1] * height, 8f, ptPaint)
                    }
                }
            }
            return bmp
        }
    }
}

