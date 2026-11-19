package com.gait.recognition

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.acos
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
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

    private val skelDim = 99
    private val skelMat: FloatArray?
    private val skelMean: FloatArray?

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

        // Cargar galería de esqueleto (gallery_skel.bin y gallery_skel_mean.bin) si están presentes
        val assetList = context.assets.list("")?.toSet() ?: emptySet()
        if ("gallery_skel.bin" in assetList && "gallery_skel_mean.bin" in assetList) {
            val skBytes = context.assets.open("gallery_skel.bin").readBytes()
            val skFb = ByteBuffer.wrap(skBytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            val sMat = FloatArray(skFb.remaining())
            skFb.get(sMat)
            skelMat = sMat

            val smBytes = context.assets.open("gallery_skel_mean.bin").readBytes()
            val smFb = ByteBuffer.wrap(smBytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            val sMean = FloatArray(smFb.remaining())
            smFb.get(sMean)
            skelMean = sMean
        } else {
            skelMat = null
            skelMean = null
        }
    }

    fun identify(
        silBuffer: List<FloatArray>,
        kptBuffer: List<Array<FloatArray>> = emptyList(),
        imgW: Int = 1280,
        imgH: Int = 720,
        alphaSil: Float = 0.65f
    ): List<MatchScore> {
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

        val probeSkelCentered = extractCenteredSkeleton(kptBuffer, imgW, imgH)

        val bestPerSubject = HashMap<String, Float>()
        val partDots = FloatArray(parts)
        val topK = 8
        val sMat = skelMat
        for (e in entries) {
            val offset = e.silIdx * silDim
            for (part in 0 until parts) {
                val start = part * channelsPerPart
                var partDot = 0f
                for (channel in 0 until channelsPerPart) {
                    val index = start + channel
                    partDot += silMat[offset + index] * normalized[index]
                }
                partDots[part] = partDot
            }
            partDots.sort()
            var sSil = 0f
            for (part in (parts - topK) until parts) {
                sSil += partDots[part]
            }
            sSil /= topK.toFloat()

            val score = if (probeSkelCentered != null && sMat != null && e.skelIdx >= 0) {
                val skOffset = e.skelIdx * skelDim
                var sSkel = 0f
                for (d in 0 until skelDim) {
                    sSkel += sMat[skOffset + d] * probeSkelCentered[d]
                }
                alphaSil * sSil + (1f - alphaSil) * sSkel
            } else {
                sSil
            }

            val prev = bestPerSubject[e.subject] ?: -1f
            if (score > prev) bestPerSubject[e.subject] = score
        }

        return bestPerSubject.entries
            .map { MatchScore(it.key, it.value) }
            .sortedByDescending { it.score }
    }

    private fun extractCenteredSkeleton(
        kptSeq: List<Array<FloatArray>>,
        imgW: Int,
        imgH: Int
    ): FloatArray? {
        val sMean = skelMean ?: return null
        if (kptSeq.size < 5) return null

        val scaleFactor = 1280f / max(1, max(imgW, imgH)).toFloat()
        val wPx = imgW * scaleFactor
        val hPx = imgH * scaleFactor

        val validFrames = kptSeq.filter { k ->
            k[5][2] >= 0.25f && k[6][2] >= 0.25f && k[11][2] >= 0.25f && k[12][2] >= 0.25f
        }
        val T = validFrames.size
        if (T < 5) return null

        val xy = Array(T) { t ->
            Array(17) { j ->
                floatArrayOf(validFrames[t][j][0] * wPx, validFrames[t][j][1] * hPx)
            }
        }

        val torsoLens = FloatArray(T)
        val midHips = Array(T) { t ->
            val hx = (xy[t][11][0] + xy[t][12][0]) / 2f
            val hy = (xy[t][11][1] + xy[t][12][1]) / 2f
            val sx = (xy[t][5][0] + xy[t][6][0]) / 2f
            val sy = (xy[t][5][1] + xy[t][6][1]) / 2f
            torsoLens[t] = hypot2(sx - hx, sy - hy)
            floatArrayOf(hx, hy)
        }
        val scale = max(percentile(torsoLens, 50f), 10f)

        val normXy = Array(T) { t ->
            Array(17) { j ->
                floatArrayOf(
                    (xy[t][j][0] - midHips[t][0]) / scale,
                    (xy[t][j][1] - midHips[t][1]) / scale
                )
            }
        }

        val feat = FloatArray(skelDim)
        var cursor = 0

        // 1. mean_pose (34) y std_pose (34)
        val stdPose = FloatArray(34)
        for (j in 0 until 17) {
            for (c in 0 until 2) {
                var sum = 0f
                for (t in 0 until T) sum += normXy[t][j][c]
                val mean = sum / T
                feat[cursor] = mean
                var varSum = 0f
                for (t in 0 until T) {
                    val d = normXy[t][j][c] - mean
                    varSum += d * d
                }
                stdPose[j * 2 + c] = sqrt(varSum / T)
                cursor++
            }
        }
        for (v in stdPose) feat[cursor++] = v

        // 2. 6 series de ángulos (rodillas, caderas, codos) -> mean(6), std(6), p10(6), p90(6) = 24
        val triplets = arrayOf(
            intArrayOf(11, 13, 15),
            intArrayOf(12, 14, 16),
            intArrayOf(5, 11, 13),
            intArrayOf(6, 12, 14),
            intArrayOf(5, 7, 9),
            intArrayOf(6, 8, 10)
        )
        val angSeries = Array(6) { aIdx ->
            val tr = triplets[aIdx]
            FloatArray(T) { t ->
                angleRad(xy[t][tr[0]], xy[t][tr[1]], xy[t][tr[2]])
            }
        }
        val angMeans = FloatArray(6) { a -> angSeries[a].average().toFloat() }
        val angStds = FloatArray(6) { a -> stdOf(angSeries[a], angMeans[a]) }
        val angP10 = FloatArray(6) { a -> percentile(angSeries[a], 10f) }
        val angP90 = FloatArray(6) { a -> percentile(angSeries[a], 90f) }
        for (v in angMeans) feat[cursor++] = v
        for (v in angStds) feat[cursor++] = v
        for (v in angP10) feat[cursor++] = v
        for (v in angP90) feat[cursor++] = v

        // 3. geom_stats (7)
        val shoulderW = FloatArray(T) { t -> dist2(normXy[t][5], normXy[t][6]) }
        val hipW = FloatArray(T) { t -> dist2(normXy[t][11], normXy[t][12]) }
        val ankleDist = FloatArray(T) { t -> dist2(normXy[t][15], normXy[t][16]) }
        val lLeg = FloatArray(T) { t -> dist2(normXy[t][11], normXy[t][13]) + dist2(normXy[t][13], normXy[t][15]) }
        val rLeg = FloatArray(T) { t -> dist2(normXy[t][12], normXy[t][14]) + dist2(normXy[t][14], normXy[t][16]) }

        val ankleMean = ankleDist.average().toFloat()
        feat[cursor++] = shoulderW.average().toFloat()
        feat[cursor++] = hipW.average().toFloat()
        feat[cursor++] = ankleMean
        feat[cursor++] = stdOf(ankleDist, ankleMean)
        feat[cursor++] = percentile(ankleDist, 95f)
        feat[cursor++] = lLeg.average().toFloat()
        feat[cursor] = rLeg.average().toFloat()

        // L2-normalizar feat crudo
        var norm1 = 0f
        for (v in feat) norm1 += v * v
        norm1 = sqrt(norm1)
        if (norm1 < 1e-8f) return null
        for (i in feat.indices) feat[i] /= norm1

        // Centrar respecto a skelMean y L2-normalizar
        var norm2 = 0f
        for (i in feat.indices) {
            feat[i] -= sMean[i]
            norm2 += feat[i] * feat[i]
        }
        norm2 = sqrt(norm2)
        if (norm2 < 1e-8f) return null
        for (i in feat.indices) feat[i] /= norm2
        return feat
    }

    private fun hypot2(dx: Float, dy: Float): Float = sqrt(dx * dx + dy * dy)
    private fun dist2(a: FloatArray, b: FloatArray): Float = hypot2(a[0] - b[0], a[1] - b[1])

    private fun angleRad(a: FloatArray, b: FloatArray, c: FloatArray): Float {
        val bax = a[0] - b[0]
        val bay = a[1] - b[1]
        val bcx = c[0] - b[0]
        val bcy = c[1] - b[1]
        val denom = (hypot2(bax, bay) * hypot2(bcx, bcy)) + 1e-6f
        val cosV = ((bax * bcx + bay * bcy) / denom).coerceIn(-1f, 1f)
        return acos(cosV)
    }

    private fun stdOf(arr: FloatArray, mean: Float): Float {
        var s = 0f
        for (v in arr) {
            val d = v - mean
            s += d * d
        }
        return sqrt(s / arr.size)
    }

    private fun percentile(arr: FloatArray, p: Float): Float {
        val sorted = arr.clone()
        sorted.sort()
        val pos = (p / 100f) * (sorted.size - 1)
        val lo = floor(pos).toInt().coerceIn(0, sorted.size - 1)
        val hi = min(sorted.size - 1, lo + 1)
        val frac = pos - lo
        return sorted[lo] * (1f - frac) + sorted[hi] * frac
    }
}
