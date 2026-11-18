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
    private val partMajorFeatures: Boolean
    private val silDim = 4096
    private val parts = 16
    private val channelsPerPart = silDim / parts
    private val silMean: FloatArray

    init {
        val modelBytes = context.assets.open("gaitbase_mobile.onnx").readBytes()
        gaitSession = env.createSession(modelBytes)

        val idxJson = context.assets.open("gallery_index.json").bufferedReader().use { it.readText() }
        val root = JSONObject(idxJson)
        partMajorFeatures = root.optString("sil_layout") == "part-major-v1"
        val arr = root.getJSONArray("entries")
        for (i in 0 until arr.length()) {
            val o = arr.getJSONObject(i)
            entries.add(GalleryEntry(o.getString("s"), o.getString("a"), o.getInt("si"), o.getInt("ki")))
        }
        require(entries.isNotEmpty()) { "gallery_index.json contains no entries" }

        val silBytes = context.assets.open("gallery_sil.bin").readBytes()
        val fb = ByteBuffer.wrap(silBytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        val rawSilMat = FloatArray(fb.remaining())
        fb.get(rawSilMat)
        require(rawSilMat.size == entries.size * silDim) {
            "gallery_sil.bin size does not match gallery_index.json"
        }

        if (partMajorFeatures) {
            val meanBytes = context.assets.open("gallery_sil_mean.bin").readBytes()
            val meanBuffer = ByteBuffer.wrap(meanBytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            silMean = FloatArray(meanBuffer.remaining())
            meanBuffer.get(silMean)
            require(silMean.size == silDim) {
                "gallery_sil_mean.bin must contain $silDim float values"
            }
            silMat = rawSilMat
        } else {
            silMean = FloatArray(silDim)
            for (entry in entries.indices) {
                val offset = entry * silDim
                for (part in 0 until parts) {
                    for (channel in 0 until channelsPerPart) {
                        silMean[part * channelsPerPart + channel] +=
                            rawSilMat[offset + channel * parts + part]
                    }
                }
            }
            for (i in silMean.indices) silMean[i] = silMean[i] / entries.size.toFloat()

            silMat = FloatArray(rawSilMat.size)
            for (entry in entries.indices) {
                val offset = entry * silDim
                for (part in 0 until parts) {
                    val partOffset = offset + part * channelsPerPart
                    var normSquared = 0f
                    for (channel in 0 until channelsPerPart) {
                        val index = part * channelsPerPart + channel
                        val value = rawSilMat[offset + channel * parts + part] - silMean[index]
                        silMat[partOffset + channel] = value
                        normSquared += value * value
                    }
                    val norm = sqrt(normSquared)
                    for (channel in 0 until channelsPerPart) {
                        val index = partOffset + channel
                        silMat[index] = if (norm > 1e-8f) silMat[index] / norm else 0f
                    }
                }
            }
        }
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

        require(emb.size == silDim) { "GaitBase ONNX output must contain $silDim values" }
        val normalized = FloatArray(silDim)
        for (part in 0 until parts) {
            val start = part * channelsPerPart
            var normSquared = 0f
            for (channel in 0 until channelsPerPart) {
                val index = start + channel
                val outputIndex = if (partMajorFeatures) index else channel * parts + part
                val value = emb[outputIndex] - silMean[index]
                normalized[index] = value
                normSquared += value * value
            }
            val norm = sqrt(normSquared)
            if (norm > 1e-8f) {
                for (channel in 0 until channelsPerPart) {
                    val index = start + channel
                    normalized[index] /= norm
                }
            } else {
                for (channel in 0 until channelsPerPart) {
                    normalized[start + channel] = 0f
                }
            }
        }

        val bestPerSubject = HashMap<String, Float>()
        for (e in entries) {
            val offset = e.silIdx * silDim
            var score = 0f
            for (part in 0 until parts) {
                val start = part * channelsPerPart
                var partDot = 0f
                for (channel in 0 until channelsPerPart) {
                    val index = start + channel
                    partDot += silMat[offset + index] * normalized[index]
                }
                score += partDot
            }
            score /= parts
            val prev = bestPerSubject[e.subject] ?: -1f
            if (score > prev) bestPerSubject[e.subject] = score
        }

        return bestPerSubject.entries
            .map { MatchScore(it.key, it.value) }
            .sortedByDescending { it.score }
    }
}

