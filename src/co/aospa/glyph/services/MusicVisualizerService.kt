/*
 * SPDX-FileCopyrightText: Paranoid Android
 * SPDX-License-Identifier: Apache-2.0
 */

package co.aospa.glyph.services

import android.app.Service
import android.content.Intent
import android.media.AudioAttributes
import android.media.AudioManager
import android.media.AudioPlaybackConfiguration
import android.media.audiofx.Visualizer
import android.os.IBinder
import co.aospa.glyph.manager.AnimationManager
import co.aospa.glyph.utils.Constants
import kotlin.math.sqrt
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch

private const val FFT_BUCKETS = 100
private const val NOISE_FLOOR = 10.0f
private const val NORMALIZE_DIVISOR = 195.0
private const val NORMALIZE_SCALE = 2.5

/** One band's FFT bucket range and its adaptive trigger/decay tuning. */
private data class BandConfig(
    val bucketStart: Int,
    val bucketCount: Int,
    val decayWindowMs: Long,
    val triggerRatio: Double,
    val triggerDecay: Double,
    val ceilingDecay: Double,
    val ceilingFloor: Double,
)

/** Nothing's AudioReactiveGlyph tuning, 5-channel config. */
private val BAND_CONFIGS = listOf(
    BandConfig(8, 4, 1000, 0.75, 0.56, 0.65, 0.5),
    BandConfig(15, 8, 500, 0.7, 0.66, 0.65, 0.5),
    BandConfig(0, 1, 300, 0.75, 0.56, 0.95, 0.75),
    BandConfig(30, 15, 200, 0.7, 0.66, 0.63, 0.5),
    BandConfig(55, 20, 100, 0.7, 0.66, 0.63, 0.5),
)

private class BandState {
    private var ceiling = 0.0
    private var lastBeatTime = 0L
    var brightness = 0.0
        private set

    fun reset() {
        ceiling = 0.0
        lastBeatTime = 0L
        brightness = 0.0
    }

    fun update(buckets: DoubleArray, config: BandConfig, now: Long) {
        val value = buckets.slice(config.bucketStart until config.bucketStart + config.bucketCount)
            .average()
        if (value > ceiling) ceiling = value

        if (ceiling > 0 && value >= ceiling * config.triggerRatio) {
            brightness = 1.0
            lastBeatTime = now
        } else {
            brightness *= config.triggerDecay
        }
        if (brightness < 0.05) brightness = 0.0

        if (now - lastBeatTime > config.decayWindowMs) {
            val decayed = ceiling * config.ceilingDecay
            if (decayed > ceiling * config.ceilingFloor) ceiling = decayed
        }
    }
}

/** The active media playback's session ID, or null if none is playing. */
private fun List<AudioPlaybackConfiguration>.activeMediaSessionId(): Int? =
    firstOrNull {
        it.audioAttributes.usage == AudioAttributes.USAGE_MEDIA && it.isActive
    }?.sessionId

/**
 * Drives the Glyph LEDs from the live output mix's FFT.
 *
 * Each band tracks its own adaptive peak, so a beat is a value clearing a fraction of that
 * band's recent ceiling rather than a fixed threshold, and brightness decays exponentially
 * between beats instead of a fixed-length flash.
 */
class MusicVisualizerService : Service() {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO.limitedParallelism(1))
    private val audioManager by lazy { getSystemService(AudioManager::class.java) }
    private var visualizer: Visualizer? = null
    private var sessionId: Int? = null

    private val bands = List(BAND_CONFIGS.size) { BandState() }
    private var wasSilent = true

    private val dataCaptureListener = object : Visualizer.OnDataCaptureListener {
        override fun onWaveFormDataCapture(
            visualizer: Visualizer,
            waveform: ByteArray,
            samplingRate: Int,
        ) {}

        override fun onFftDataCapture(visualizer: Visualizer, fft: ByteArray, samplingRate: Int) {
            if (visualizer !== this@MusicVisualizerService.visualizer) return
            processFft(fft)
        }
    }

    private val playbackCallback = object : AudioManager.AudioPlaybackCallback() {
        override fun onPlaybackConfigChanged(configs: MutableList<AudioPlaybackConfiguration>) {
            updateVisualizer(configs.activeMediaSessionId())
        }
    }

    override fun onCreate() {
        audioManager.registerAudioPlaybackCallback(playbackCallback, null)
        updateVisualizer(audioManager.activePlaybackConfigurations.activeMediaSessionId())
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        return START_STICKY
    }

    override fun onDestroy() {
        audioManager.unregisterAudioPlaybackCallback(playbackCallback)
        releaseVisualizer()
        scope.cancel()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun updateVisualizer(activeSessionId: Int?) {
        scope.launch {
            if (activeSessionId == null) {
                releaseVisualizer()
            } else {
                attachVisualizer(activeSessionId)
            }
        }
    }

    private fun attachVisualizer(newSessionId: Int) {
        if (newSessionId == sessionId && visualizer != null) return
        releaseVisualizer()
        sessionId = newSessionId
        try {
            val newVisualizer = Visualizer(newSessionId)
            visualizer = newVisualizer
            newVisualizer.apply {
                setServerDiedListener {
                    if (this@MusicVisualizerService.visualizer === newVisualizer) releaseVisualizer()
                }
                captureSize = Visualizer.getCaptureSizeRange()[1]
                scalingMode = Visualizer.SCALING_MODE_AS_PLAYED
                setDataCaptureListener(dataCaptureListener, Visualizer.getMaxCaptureRate(), false, true)
                enabled = true
            }
        } catch (_: RuntimeException) {
            releaseVisualizer()
        }
    }

    private fun releaseVisualizer() {
        sessionId = null
        val releasedVisualizer = visualizer
        visualizer = null
        bands.forEach { it.reset() }
        wasSilent = true
        AnimationManager.updateLedFrame(IntArray(bands.size))
        try {
            releasedVisualizer?.apply {
                enabled = false
                release()
            }
        } catch (_: RuntimeException) {
            // The audio server may already be unavailable.
        }
    }

    private fun processFft(fft: ByteArray) {
        if (fft.all { it == 0.toByte() }) {
            if (!wasSilent) {
                bands.forEach { it.reset() }
                wasSilent = true
                writeLedFrame(IntArray(bands.size))
            }
            return
        }
        wasSilent = false

        val buckets = bucketMagnitudes(fft)
        val now = System.currentTimeMillis()
        bands.forEachIndexed { i, state -> state.update(buckets, BAND_CONFIGS[i], now) }

        val maxBrightness = Constants.getMaxBrightness()
        writeLedFrame(IntArray(bands.size) { i -> (bands[i].brightness * maxBrightness).toInt() })
    }

    private fun writeLedFrame(pattern: IntArray) {
        scope.launch { AnimationManager.updateLedFrame(pattern) }
    }

    private fun bucketMagnitudes(fft: ByteArray): DoubleArray {
        val numComplexBins = (fft.size / 2) - 1
        val binsPerBucket = numComplexBins / FFT_BUCKETS
        return DoubleArray(FFT_BUCKETS) { i ->
            var sum = 0f
            for (j in 0 until binsPerBucket) {
                val bin = 1 + (i * binsPerBucket) + j
                val idx = bin * 2
                val re = fft[idx].toFloat()
                val im = fft[idx + 1].toFloat()
                sum += (sqrt(re * re + im * im) - NOISE_FLOOR).coerceAtLeast(0f)
            }
            ((sum / binsPerBucket).coerceAtLeast(0f) / NORMALIZE_DIVISOR) * NORMALIZE_SCALE
        }
    }
}
