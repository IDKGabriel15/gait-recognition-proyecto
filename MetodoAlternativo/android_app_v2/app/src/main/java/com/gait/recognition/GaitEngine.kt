package com.gait.recognition

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

data class GalleryEntry(val subject: String, val angle: String, val silIdx: Int)
data class MatchScore(val subject: String, val score: Float)

/**
 * Motor v2 (Método Alternativo) 100% Offline:
 *   - Backbone: DeepGaitV2-P3D (convoluciones temporales Pseudo-3D, ONNX FP32).
 *   - Submuestreo uniforme a 24 frames (~10 fps) para alinear tasa temporal con la galería.
 *   - TTA Horizontal Flip: evalúa secuencia original y espejada horizontalmente.
 *   - Adaptación de dominio: Part-Wise PCA-Whitening (16x256) + centrado y L2 por parte.
 *   - Matching: Top-10 franjas simétricas + Top-3 Gallery Mean por sujeto (sin sesgo de ángulo).
 */
class GaitEngine(context: Context, private val env: OrtEnvironment) {
    private val gaitSession: OrtSession
    private val entries = mutableListOf<GalleryEntry>()
    private val silMat: FloatArray
    private val silMean: FloatArray
    private val pcaMeans: FloatArray
    private val pcaProj: FloatArray

    private val parts = 16
    private val channelsPerPart = 256
    private val silDim = parts * channelsPerPart
    private val topKParts: Int
    private val topKGallery: Int

    init {
        val opts = OrtSession.SessionOptions()
        opts.setIntraOpNumThreads(4)
        val modelBytes = context.assets.open("deepgaitv2_fp32.onnx").readBytes()
        gaitSession = env.createSession(modelBytes, opts)

        val idxJson = context.assets.open("gallery_v2_index.json").bufferedReader().use { it.readText() }
        val root = JSONObject(idxJson)
        topKParts = root.optInt("top_k_parts", 10)
        topKGallery = root.optInt("top_k_gallery", 3)
        val arr = root.getJSONArray("entries")
        for (i in 0 until arr.length()) {
            val o = arr.getJSONObject(i)
            entries.add(GalleryEntry(o.getString("s"), o.getString("a"), o.getInt("si")))
        }
        require(entries.isNotEmpty()) { "gallery_v2_index.json está vacío" }

        silMat = readFloatAsset(context, "gallery_v2_sil.bin")
        silMean = readFloatAsset(context, "gallery_v2_sil_mean.bin")
        pcaMeans = readFloatAsset(context, "pca_v2_means.bin")
        pcaProj = readFloatAsset(context, "pca_v2_proj.bin")

        require(silMat.size == entries.size * silDim) { "Tamaño inválido en gallery_v2_sil.bin" }
        require(silMean.size == silDim) { "Tamaño inválido en gallery_v2_sil_mean.bin" }
        require(pcaMeans.size == silDim) { "Tamaño inválido en pca_v2_means.bin" }
        require(pcaProj.size == parts * channelsPerPart * channelsPerPart) { "Tamaño inválido en pca_v2_proj.bin" }
    }

    private fun readFloatAsset(context: Context, assetName: String): FloatArray {
        val bytes = context.assets.open(assetName).readBytes()
        val fb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        val out = FloatArray(fb.remaining())
        fb.get(out)
        return out
    }

    private fun uniformSubsample(seq: List<FloatArray>, maxFrames: Int = 24): List<FloatArray> {
        if (seq.size <= maxFrames) return seq
        val out = ArrayList<FloatArray>(maxFrames)
        val step = (seq.size - 1).toFloat() / (maxFrames - 1).toFloat()
        for (i in 0 until maxFrames) {
            val idx = (i * step).roundToInt().coerceIn(0, seq.size - 1)
            out.add(seq[idx])
        }
        return out
    }

    private fun flipHorizontal(seq: List<FloatArray>): List<FloatArray> {
        val h = 64
        val w = 44
        return seq.map { frame ->
            val flipped = FloatArray(h * w)
            for (y in 0 until h) {
                val rowOff = y * w
                for (x in 0 until w) {
                    flipped[rowOff + x] = frame[rowOff + (w - 1 - x)]
                }
            }
            flipped
        }
    }

    private fun runDeepGaitV2(seq: List<FloatArray>): FloatArray {
        val seqLen = seq.size
        val flat = FloatBuffer.allocate(seqLen * 64 * 44)
        for (frameSil in seq) {
            flat.put(frameSil)
        }
        flat.rewind()

        val tensor = OnnxTensor.createTensor(
            env,
            flat,
            longArrayOf(1L, 1L, seqLen.toLong(), 64L, 44L)
        )

        val rawEmb: FloatArray
        tensor.use { t ->
            val out = gaitSession.run(mapOf(gaitSession.inputNames.first() to t))
            out.use { res ->
                @Suppress("UNCHECKED_CAST")
                rawEmb = (res[0].value as Array<FloatArray>)[0]
            }
        }
        return transformAndNormalize(rawEmb)
    }

    /** Aplica Part-Wise PCA-Whitening -> resta media de galería -> L2 por cada una de las 16 franjas. */
    private fun transformAndNormalize(rawEmb: FloatArray): FloatArray {
        val out = FloatArray(silDim)
        val centeredIn = FloatArray(channelsPerPart)
        for (part in 0 until parts) {
            val pOff = part * channelsPerPart
            for (c in 0 until channelsPerPart) {
                centeredIn[c] = rawEmb[pOff + c] - pcaMeans[pOff + c]
            }
            val projOff = part * channelsPerPart * channelsPerPart
            var normSq = 0f
            for (outC in 0 until channelsPerPart) {
                var acc = 0f
                for (inC in 0 until channelsPerPart) {
                    acc += centeredIn[inC] * pcaProj[projOff + inC * channelsPerPart + outC]
                }
                val v = acc - silMean[pOff + outC]
                out[pOff + outC] = v
                normSq += v * v
            }
            val norm = sqrt(normSq)
            if (norm > 1e-8f) {
                for (c in 0 until channelsPerPart) {
                    out[pOff + c] = out[pOff + c] / norm
                }
            } else {
                for (c in 0 until channelsPerPart) {
                    out[pOff + c] = 0f
                }
            }
        }
        return out
    }

    private fun topKPartSim(probe: FloatArray, galOffset: Int, partDots: FloatArray): Float {
        for (part in 0 until parts) {
            val start = part * channelsPerPart
            var dot = 0f
            for (c in 0 until channelsPerPart) {
                val idx = start + c
                dot += silMat[galOffset + idx] * probe[idx]
            }
            partDots[part] = dot
        }
        partDots.sort()
        var sum = 0f
        for (p in (parts - topKParts) until parts) {
            sum += partDots[p]
        }
        return sum / topKParts.toFloat()
    }

    fun identify(
        silBuffer: List<FloatArray>,
        useTtaFlip: Boolean = true
    ): List<MatchScore> {
        if (silBuffer.size < 8) return emptyList()

        // 1. Submuestreo uniforme a 24 frames (~10 fps) para alinear tasa temporal P3D
        val subSeq = uniformSubsample(silBuffer, maxFrames = 24)
        val probeOrig = runDeepGaitV2(subSeq)
        val probeFlip = if (useTtaFlip) runDeepGaitV2(flipHorizontal(subSeq)) else null

        // 2. Acumular todas las similitudes por sujeto (mejor entre original y espejado por entrada)
        val scoresBySubject = HashMap<String, MutableList<Float>>()
        val partDots = FloatArray(parts)

        for (e in entries) {
            val offset = e.silIdx * silDim
            var sim = topKPartSim(probeOrig, offset, partDots)
            if (probeFlip != null) {
                val simFlip = topKPartSim(probeFlip, offset, partDots)
                if (simFlip > sim) sim = simFlip
            }
            val list = scoresBySubject.getOrPut(e.subject) { ArrayList() }
            list.add(sim)
        }

        // 3. Top-K Gallery Mean por sujeto (K = 3)
        val results = ArrayList<MatchScore>(scoresBySubject.size)
        for ((subj, list) in scoresBySubject) {
            list.sortDescending()
            val k = min(topKGallery, list.size)
            var sum = 0f
            for (i in 0 until k) sum += list[i]
            results.add(MatchScore(subj, sum / k.toFloat()))
        }

        results.sortByDescending { it.score }
        return results
    }
}
