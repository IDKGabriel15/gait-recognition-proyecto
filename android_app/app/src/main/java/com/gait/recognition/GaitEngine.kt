package com.gait.recognition

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.sqrt

data class GalleryEntry(val subject: String, val angle: String, val silIdx: Int, val skelIdx: Int)
data class MatchScore(val subject: String, val score: Float)

class GaitEngine(context: Context, private val env: OrtEnvironment) {
    private val gaitSession: OrtSession
    private val entries = mutableListOf<GalleryEntry>()
    private val silMat: FloatArray
    private val silDim = 4096

    init {
        val modelBytes = context.assets.open("gaitbase_mobile.onnx").readBytes()
        gaitSession = env.createSession(modelBytes)

        val idxJson = context.assets.open("gallery_index.json").bufferedReader().use { it.readText() }
        val root = JSONObject(idxJson)
        val arr = root.getJSONArray("entries")
        for (i in 0 until arr.length()) {
            val o = arr.getJSONObject(i)
            entries.add(GalleryEntry(o.getString("s"), o.getString("a"), o.getInt("si"), o.getInt("ki")))
        }

        val silBytes = context.assets.open("gallery_sil.bin").readBytes()
        val fb = ByteBuffer.wrap(silBytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        silMat = FloatArray(fb.remaining())
        fb.get(silMat)
    }

    fun identify(silBuffer: List<FloatArray>): List<MatchScore> {
        val seqLen = silBuffer.size
        if (seqLen < 8) return emptyList()

        val flat = FloatBuffer.allocate(seqLen * 64 * 44)
        for (frameSil in silBuffer) {
            flat.put(frameSil)
        }
        flat.rewind()

        val tensor = OnnxTensor.createTensor(
            env,
            flat,
            longArrayOf(1L, 1L, seqLen.toLong(), 64L, 44L)
        )

        val emb: FloatArray
        tensor.use { t ->
            val out = gaitSession.run(mapOf(gaitSession.inputNames.first() to t))
            out.use { res ->
                @Suppress("UNCHECKED_CAST")
                emb = (res[0].value as Array<FloatArray>)[0]
            }
        }

        // L2-normalizar por seguridad
        var norm = 0f
        for (v in emb) norm += v * v
        norm = sqrt(norm).coerceAtLeast(1e-8f)
        for (i in emb.indices) emb[i] /= norm

        val bestPerSubject = HashMap<String, Float>()
        for (e in entries) {
            val offset = e.silIdx * silDim
            var dot = 0f
            for (d in 0 until silDim) {
                dot += silMat[offset + d] * emb[d]
            }
            val prev = bestPerSubject[e.subject] ?: -1f
            if (dot > prev) bestPerSubject[e.subject] = dot
        }

        return bestPerSubject.entries
            .map { MatchScore(it.key, it.value) }
            .sortedByDescending { it.score }
    }
}

